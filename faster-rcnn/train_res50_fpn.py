import os
import datetime
import warnings
import numpy as np
import torch
from torch.cuda.amp import autocast, GradScaler
import transforms
from network_files import FasterRCNN, FastRCNNPredictor
from backbone import resnet50_fpn_backbone
from my_dataset import ClusterDataSet
from train_utils import GroupedBatchSampler, create_aspect_ratio_groups
from train_utils import train_eval_utils as utils



CLASS_NAMES = ["U-type", "inverted-U", "chain", "special"]  # 示例：4类，对应num_classes=4

def create_model(num_classes, load_pretrain_weights=True):
    """创建Faster R-CNN模型"""
    backbone = resnet50_fpn_backbone(
        pretrain_path="./backbone/resnet50.pth",
        norm_layer=torch.nn.BatchNorm2d,
        trainable_layers=5
    )
    model = FasterRCNN(backbone=backbone, num_classes=91)

    if load_pretrain_weights:
        weights_dict = torch.load("./backbone/fasterrcnn_resnet50_fpn_coco.pth", map_location='cpu')
        missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
        if missing_keys or unexpected_keys:
            warnings.warn(f"Missing keys: {missing_keys}\nUnexpected keys: {unexpected_keys}")

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def main(args):
    # 初始化设置
    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using {device.type} device training.")

    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    # 1. 主结果文件（含整体指标）
    results_file = os.path.join(args.output_dir, f"results_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.txt")
    # 2. 新增：每类AP+平均mAP的汇总文件
    per_class_results_file = os.path.join(args.output_dir, f"per_class_map_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.txt")

    # 数据预处理
    # transforms.RandomPerspective(0.0, 0.1, 0.5, 0.0),

    data_transform = {
        "train": transforms.Compose([
            transforms.RandomPerspective(0.0, 0.1, 0.5, 0.0),
            transforms.RandomHSV(hgain=0.015, sgain=0.7, vgain=0.4),
            transforms.RandomFlip(p=0.5, direction="horizontal"),
            transforms.ToTensor(),

        ]),
        "val": transforms.Compose([

            transforms.ToTensor(),

        ])
    }
    # 检查数据集路径
    if not os.path.exists(os.path.join(args.data_path, "pic")):
        raise FileNotFoundError(f"Dataset not found in path: {args.data_path}")

    # 数据加载
    def create_data_loader(dataset, batch_size, shuffle):
        if os.name == 'nt':
            num_workers = 0
            multiprocessing_context = None
        else:
            num_workers = min(4, os.cpu_count())
            multiprocessing_context = 'spawn'

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=num_workers,
            persistent_workers=num_workers>0,
            collate_fn=dataset.collate_fn,
            multiprocessing_context=multiprocessing_context
        )

    # 训练集
    print(args.data_path)
    train_dataset = ClusterDataSet(args.data_path, data_transform["train"], "train.txt")
    if args.aspect_ratio_group_factor >= 0:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        group_ids = create_aspect_ratio_groups(train_dataset, k=args.aspect_ratio_group_factor)
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_sampler=train_batch_sampler,
            pin_memory=True,
            num_workers=0,
            persistent_workers=False,
            collate_fn=train_dataset.collate_fn,
        )
    else:
        train_loader = create_data_loader(train_dataset, args.batch_size, True)

    # 验证集
    val_dataset = ClusterDataSet(args.data_path, data_transform["val"], "val.txt")
    val_loader = create_data_loader(val_dataset, 1, False)

    # 初始化模型
    model = create_model(num_classes=args.num_classes+1)
    model.to(device)

    # 优化器设置
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    # 混合精度训练
    scaler = GradScaler() if (args.amp and torch.cuda.is_bf16_supported()) else None
    if args.amp and not torch.cuda.is_bf16_supported():
        warnings.warn("AMP is not supported on this GPU, disabled automatically")

    # 学习率调度器
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.33)
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=args.lr,  # 从完整学习率开始
        end_factor=0.01,  # 衰减到初始学习率的1%
        total_iters=args.epochs  # 衰减持续剩余epoch
    )
    # 恢复训练
    best_map = 0.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        best_map = checkpoint.get('best_map', 0.0)
        if args.amp and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        print(f"Resuming training from epoch {args.start_epoch}, best mAP: {best_map:.4f}")

    # ===================== 新增：初始化记录列表 =====================
    train_loss = []
    learning_rate = []
    val_map = []  # 平均mAP
    per_class_ap_history = {cls_name: [] for cls_name in CLASS_NAMES}  # 每类AP历史

    # 训练循环
    for epoch in range(args.start_epoch, args.epochs):
        # 训练阶段
        mean_loss, lr = utils.train_one_epoch(
            model, optimizer, train_loader,
            device=device, epoch=epoch,
            print_freq=100, warmup=True,
            scaler=scaler
        )
        train_loss.append(mean_loss.item())
        learning_rate.append(lr)

        # 更新学习率
        lr_scheduler.step()

        coco_info, per_class_ap = utils.evaluate(
            model, val_loader, device=device,
            num_classes=args.num_classes+1,  # 含背景类
            class_names=CLASS_NAMES         # 传入类别名
        )
        print(per_class_ap)
        # 计算平均mAP（所有类别AP的均值）
        if per_class_ap:
            current_map = np.mean(list(per_class_ap.values()))  # 平均mAP
            # 记录每类AP到历史列表
            for cls_name in CLASS_NAMES:
                per_class_ap_history[cls_name].append(per_class_ap.get(cls_name, 0.0))
        else:
            current_map = 0.0
            # 无结果时填充0
            for cls_name in CLASS_NAMES:
                per_class_ap_history[cls_name].append(0.0)
        val_map.append(current_map)

        # ===================== 1. 写入整体结果文件（原有逻辑） =====================
        with open(results_file, "a") as f:
            result_info = [f"{i:.4f}" for i in coco_info + [mean_loss.item()]] + [f"{lr:.6f}"]
            f.write(f"epoch:{epoch} {'  '.join(result_info)}\n")

        # ===================== 2. 写入每类AP+平均mAP汇总文件 =====================
        with open(per_class_results_file, "a", encoding="utf-8") as f:
            # 行格式：epoch | 类别1 AP | 类别2 AP | 类别3 AP | 类别4 AP | 平均mAP
            ap_str = "\t".join([f"{per_class_ap.get(cls, 0.0):.4f}" for cls in CLASS_NAMES])
            f.write(f"epoch:{epoch}\t{ap_str}\t平均mAP:{current_map:.4f}\n")

        # 打印本轮每类AP和平均mAP（控制台可视化）
        print(f"\n===== Epoch {epoch} 类别AP结果 =====")
        for cls_name, ap in per_class_ap.items():
            print(f"{cls_name}: {ap:.4f}")
        print(f"平均mAP: {current_map:.4f}\n")

        # 模型保存逻辑（原有逻辑不变）
        def save_checkpoint(epoch, is_best=False):
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'best_map': best_map,
                'args': vars(args),
                'per_class_ap_history': per_class_ap_history,  # 新增：保存每类AP历史
                'val_map': val_map  # 新增：保存平均mAP历史
            }
            if scaler is not None:
                checkpoint['scaler'] = scaler.state_dict()

            torch.save(
                checkpoint,
                os.path.join(args.output_dir, "best_model.pth" if is_best else "latest_model.pth"),
                _use_new_zipfile_serialization=True
            )

        # 保存最佳模型
        if current_map > best_map:
            best_map = current_map
            save_checkpoint(epoch, is_best=True)
            print(f"New best model saved with mAP: {best_map:.4f}")

        # 保存最新模型
        save_checkpoint(epoch)
        if epoch % 10 == 0:
            torch.save(
                model.state_dict(),
                os.path.join(args.output_dir, f"epoch_{epoch}_model.pth")
            )

        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ===================== 新增：绘制每类AP+平均mAP曲线 =====================
    try:
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["SimHei"]  # 解决中文显示
        plt.rcParams["axes.unicode_minus"] = False    # 解决负号显示

        # 创建画布
        plt.figure(figsize=(12, 6))
        epochs = list(range(args.start_epoch, args.epochs))

        # 绘制每类AP曲线
        colors = ["red", "blue", "green", "orange"]  # 4类对应4种颜色，可扩展
        for idx, cls_name in enumerate(CLASS_NAMES):
            plt.plot(epochs, per_class_ap_history[cls_name],
                     label=f"{cls_name} AP", color=colors[idx], marker="o", markersize=2)

        # 绘制平均mAP曲线（加粗）
        plt.plot(epochs, val_map, label="平均mAP", color="black", linewidth=2, marker="s", markersize=3)

        # 图表配置
        plt.xlabel("Epoch")
        plt.ylabel("AP/mAP (%)" if max(val_map) <= 100 else "AP/mAP")
        plt.title("各类别AP及平均mAP变化曲线")
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()

        # 保存曲线图片
        curve_path = os.path.join(args.output_dir, "per_class_ap_curve.png")
        plt.savefig(curve_path, dpi=300)
        print(f"\n类别AP曲线已保存至：{curve_path}")
        plt.show()

    except ImportError:
        warnings.warn("matplotlib未安装，跳过类别AP曲线绘制")
    except Exception as e:
        warnings.warn(f"绘制类别AP曲线失败：{str(e)}")

    # 原有可视化（损失+学习率）
    if train_loss and learning_rate:
        try:
            from plot_curve import plot_loss_and_lr
            plot_loss_and_lr(train_loss, learning_rate)
        except ImportError:
            warnings.warn("plot_curve module not found, skipping loss/lr visualization")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0', help='device')
    parser.add_argument('--data-path', default=r'D:\project_SJ\deep-learning-for-image-processing-master\pytorch_object_detection\faster_rcnn\ClusterDataset', help='dataset root')
    parser.add_argument('--num-classes', default=4, type=int, help='number of classes (excluding background)')
    parser.add_argument('--output-dir', default=r'./save_weight_tuning', help='path to save checkpoints')
    parser.add_argument('--resume', default=r'', help='checkpoint to resume from')
    parser.add_argument('--start-epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--epochs', default=120, type=int, help='total training epochs')
    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, dest='weight_decay')
    parser.add_argument('--batch-size', default=12, type=int, help='batch size')
    parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)
    parser.add_argument("--amp", action="store_true", help="use mixed precision training")
    args = parser.parse_args()
    print(args)

    main(args)