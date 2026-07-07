# 不用导入torchutils,将所有功能都集成到这里，增加了各类别的精度
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import os
import os.path as osp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


class MetricMonitor:
    def __init__(self):
        self.metrics = {}

    def update(self, name, value):
        if name not in self.metrics:
            self.metrics[name] = {"val": 0, "count": 0, "avg": 0}
        self.metrics[name]["val"] += value
        self.metrics[name]["count"] += 1
        self.metrics[name]["avg"] = self.metrics[name]["val"] / self.metrics[name]["count"]

    def __str__(self):
        return " | ".join([f"{name}: {metrics['avg']:.4f}" for name, metrics in self.metrics.items()])


def accuracy(output, target):
    _, preds = torch.max(output, 1)
    correct = torch.eq(preds, target).sum().item()
    return correct / len(target)


def adjust_learning_rate(optimizer, epoch, params, iter, nBatch):
    lr = params['lr']
    # 简单的余弦退火学习率调整（可根据需求修改）
    lr *= 0.5 * (1 + np.cos(np.pi * (epoch * nBatch + iter) / (params['epochs'] * nBatch)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def get_torch_transforms(img_size=224):
    """数据预处理（与原代码torchutils保持一致）"""
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    return data_transforms

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
print(f'Using device: {device}')

# 固定随机种子
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

# 数据集路径
data_path = r"D:\project_SJ\Resnet50d_classification\datasets_classify\cluster_four_origin.v10"

# 超参数设置
params = {
    'model': 'resnet50d',
    "img_size": 224,
    'device': device,
    "name": "_origin_v10",
    'lr': 1e-4,
    'batch_size': 128,
    'num_workers': 0,
    'epochs': 200,
    "save_dir": "./runs/",
    "pretrained": False,
    "num_classes": len(os.listdir(data_path)),
    'weight_decay': 1e-5,
    'max_grad_norm': 1.0,
    'seed': seed
}

# 获取类别名称（用于可视化标注）
class_names = sorted(os.listdir(data_path))
print(f"类别列表: {class_names}")


# 定义模型
class SELFMODEL(nn.Module):
    def __init__(self, model_name=params['model'], out_features=params['num_classes'],
                 pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        if model_name[:3] == "res":
            n_features = self.model.fc.in_features
            self.model.fc = nn.Linear(n_features, out_features)
        elif model_name[:3] == "vit":
            n_features = self.model.head.in_features
            self.model.head = nn.Linear(n_features, out_features)
        else:
            n_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(n_features, out_features)

    def forward(self, x):
        x = self.model(x)
        return x


def train(train_loader, model, criterion, optimizer, epoch, params):
    metric_monitor = MetricMonitor()
    model.train()
    nBatch = len(train_loader)
    stream = tqdm(train_loader)
    for i, (images, target) in enumerate(stream, start=1):
        images = images.to(params['device'], non_blocking=True)
        target = target.to(params['device'], non_blocking=True)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            output = model(images)
            loss = criterion(output, target.long())
        acc = accuracy(output, target)
        metric_monitor.update('Loss', loss.item())
        metric_monitor.update('Accuracy', acc)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), params['max_grad_norm'])
        optimizer.step()
        lr = adjust_learning_rate(optimizer, epoch, params, i, nBatch)
        stream.set_description(
            "Epoch: {epoch}. LR:{lr:.5e}. Train.      {metric_monitor}".format(
                epoch=epoch, lr=lr, metric_monitor=metric_monitor)
        )
    return metric_monitor.metrics['Accuracy']["avg"], metric_monitor.metrics['Loss']["avg"]


def validate(val_loader, model, criterion, epoch, params, class_names):
    metric_monitor = MetricMonitor()
    model.eval()
    stream = tqdm(val_loader)

    # 存储所有预测和真实标签（用于计算每类指标）
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for i, (images, target) in enumerate(stream, start=1):
            images = images.to(params['device'], non_blocking=True)
            target = target.to(params['device'], non_blocking=True)
            output = model(images)
            loss = criterion(output, target.long())
            acc = accuracy(output, target)

            # 收集预测结果和真实标签
            preds = torch.argmax(output, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

            metric_monitor.update('Loss', loss.item())
            metric_monitor.update('Accuracy', acc)
            stream.set_description(
                "Epoch: {epoch}. Validation. {metric_monitor}".format(
                    epoch=epoch, metric_monitor=metric_monitor)
            )

    # 计算每类精度、召回率、F1
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # 处理单类别边界情况
    if len(np.unique(all_targets)) == 1:
        class_precision = {class_names[0]: 1.0}
        class_recall = {class_names[0]: 1.0}
        class_f1 = {class_names[0]: 1.0}
    else:
        class_precision = precision_score(all_targets, all_preds, average=None, zero_division=0)
        class_recall = recall_score(all_targets, all_preds, average=None, zero_division=0)
        class_f1 = f1_score(all_targets, all_preds, average=None, zero_division=0)

        # 转换为字典（类别名称: 指标值）
        class_precision = {class_names[i]: class_precision[i] for i in range(len(class_names))}
        class_recall = {class_names[i]: class_recall[i] for i in range(len(class_names))}
        class_f1 = {class_names[i]: class_f1[i] for i in range(len(class_names))}

    # 打印每类指标（可选）
    if epoch % 10 == 0:  # 每10个epoch打印一次
        print(f"\nEpoch {epoch} 每类精度: {class_precision}")

    return (metric_monitor.metrics['Accuracy']["avg"],
            metric_monitor.metrics['Loss']["avg"],
            class_precision, class_recall, class_f1)


def show_loss_acc(acc, loss, val_acc, val_loss, save_dir):
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.savefig(save_dir, dpi=100)
    plt.close()


def show_class_precision_trend(class_precision_history, save_dir, fold):
    """
    class_precision_history: 列表，每个元素是{类别名: 精度值}（对应每个epoch）
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(class_precision_history) + 1)

    # 提取每个类别的精度序列
    for cls_name in class_names:
        cls_precisions = [epoch_data[cls_name] for epoch_data in class_precision_history]
        plt.plot(epochs, cls_precisions, label=cls_name, marker='.')

    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.title(f'Fold {fold + 1} - Class Precision Trend')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.ylim(0, 1.05)
    plt.savefig(osp.join(save_dir, f'class_precision_trend_fold{fold}.png'), dpi=100)
    plt.close()


def show_class_precision_summary(fold_results, save_dir):
    """
    生成每类精度的柱状图（每折最终精度 + 平均值）
    """
    # 整理数据：{类别名: [fold1精度, fold2精度, ...]}
    cls_precision_dict = {cls: [] for cls in class_names}
    for fold_detail in fold_results['fold_details']:
        final_cls_precision = fold_detail['final_class_precision']
        for cls in class_names:
            cls_precision_dict[cls].append(final_cls_precision[cls])

    # 计算每类平均精度
    cls_avg_precision = {cls: np.mean(precisions) for cls, precisions in cls_precision_dict.items()}

    # 绘图
    plt.figure(figsize=(12, 6))
    x = np.arange(len(class_names))
    width = 0.2
    num_folds = len(fold_results['fold_details'])

    # 绘制每折的精度
    for fold in range(num_folds):
        fold_precisions = [cls_precision_dict[cls][fold] for cls in class_names]
        plt.bar(x + (fold - num_folds / 2 + 0.5) * width, fold_precisions, width, label=f'Fold {fold + 1}')

    # 绘制平均值
    avg_precisions = [cls_avg_precision[cls] for cls in class_names]
    plt.bar(x + width * num_folds / 2, avg_precisions, width, label='Average', color='black', alpha=0.7)

    plt.xlabel('Class')
    plt.ylabel('Precision')
    plt.title('Class Precision Summary (Final Epoch)')
    plt.xticks(x, class_names)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(alpha=0.3, axis='y')
    plt.savefig(osp.join(save_dir, 'class_precision_summary.png'), dpi=100)
    plt.close()


# 扩展K-Fold训练：新增每类精度记录
def train_k_fold(params, data_transforms, num_folds=5):
    full_dataset = datasets.ImageFolder(data_path, data_transforms['train'])
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=params.get('seed', 42))
    fold_results = {
        'best_acc': [],
        'final_acc': [],
        'history': [],
        "fold_details": [],
        'all_class_precision': []  # 新增：存储所有折的每类精度
    }

    fold_dir = osp.join(params['save_dir'], f"k_fold{params['name']}")
    os.makedirs(fold_dir, exist_ok=True)
    csv_header = "fold,epoch,train_acc,train_loss,val_acc,val_loss," + ",".join(
        [f"{cls}_precision" for cls in class_names]) + "\n"
    csv_path = osp.join(fold_dir, "results.csv")
    with open(csv_path, 'w') as f:
        f.write(csv_header)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(full_dataset)):
        print(f"\n===== Fold {fold + 1}/{num_folds} =====")
        # 划分数据集
        train_subset = Subset(full_dataset, train_idx)
        val_subset = Subset(full_dataset, val_idx)

        # 注意：验证集使用val_transforms
        val_dataset = datasets.ImageFolder(data_path, data_transforms['val'])
        val_subset = Subset(val_dataset, val_idx)

        train_loader = DataLoader(
            train_subset, batch_size=params['batch_size'], shuffle=True,
            num_workers=params['num_workers'], pin_memory=True
        )
        val_loader = DataLoader(
            val_subset, batch_size=params['batch_size'], shuffle=False,
            num_workers=params['num_workers'], pin_memory=True
        )

        # 初始化模型
        model = SELFMODEL(
            model_name=params['model'],
            out_features=params['num_classes'],
            pretrained=params['pretrained']
        ).to(params['device'])
        criterion = nn.CrossEntropyLoss().to(params['device'])
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=params['lr'],
            weight_decay=params['weight_decay']
        )

        # 存储日志
        accs, losss, val_accs, val_losss = [], [], [], []
        class_precision_history = []  # 每epoch的每类精度
        best_acc = 0.0
        fold_save_dir = osp.join(fold_dir, f"_fold_{fold}")
        os.makedirs(fold_save_dir, exist_ok=True)

        for epoch in range(1, params['epochs'] + 1):
            # 训练
            acc, loss = train(train_loader, model, criterion, optimizer, epoch, params)
            # 验证（新增每类精度返回值）
            val_acc, val_loss, class_precision, _, _ = validate(val_loader, model, criterion, epoch, params,
                                                                class_names)

            # 记录日志
            accs.append(acc)
            losss.append(loss)
            val_accs.append(val_acc)
            val_losss.append(val_loss)
            class_precision_history.append(class_precision)

            # 保存到CSV（含每类精度）
            cls_precision_vals = [class_precision[cls] for cls in class_names]
            with open(csv_path, 'a') as f:
                f.write(f"{fold},{epoch},{acc:.4f},{loss:.4f},{val_acc:.4f},{val_loss:.4f},")
                f.write(",".join([f"{p:.4f}" for p in cls_precision_vals]) + "\n")

            # 保存模型
            if val_acc >= best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), osp.join(fold_save_dir, f"best_model_fold{fold}.pth"))
            torch.save(model.state_dict(), osp.join(fold_save_dir, f"latest_model_fold{fold}.pth"))

        # 生成当前折的可视化
        show_loss_acc(accs, losss, val_accs, val_losss, osp.join(fold_save_dir, f'training_curve_fold{fold}.png'))
        show_class_precision_trend(class_precision_history, fold_save_dir, fold)

        # 记录当前折结果（新增每类精度）
        final_class_precision = class_precision_history[-1]
        fold_results['best_acc'].append(best_acc)
        fold_results['final_acc'].append(val_accs[-1])
        fold_results['history'].append((accs, losss, val_accs, val_losss))
        fold_results['fold_details'].append({
            'fold': fold,
            'best_acc': best_acc,
            'final_acc': val_accs[-1],
            'train_acc': accs,
            'val_acc': val_accs,
            'final_class_precision': final_class_precision  # 新增
        })
        fold_results['all_class_precision'].append(class_precision_history)

    # 生成所有折的每类精度汇总图
    show_class_precision_summary(fold_results, fold_dir)

    # 打印汇总结果
    print("\n===== K-Fold 交叉验证结果 =====")
    print(f"每折最佳验证准确率: {fold_results['best_acc']}")
    print(f"平均最佳准确率: {sum(fold_results['best_acc']) / num_folds:.4f}")
    print(f"每折最终验证准确率: {fold_results['final_acc']}")
    print(f"平均最终准确率: {sum(fold_results['final_acc']) / num_folds:.4f}")
    # 打印每类平均精度
    print("\n===== 每类平均精度（所有折最终epoch） =====")
    for cls in class_names:
        cls_precisions = [fold_detail['final_class_precision'][cls] for fold_detail in fold_results['fold_details']]
        print(f"{cls}: {np.mean(cls_precisions):.4f} (±{np.std(cls_precisions):.4f})")
    # 保存汇总CSV
    summary_df = pd.DataFrame({
        'fold': range(num_folds),
        'best_acc': fold_results['best_acc'],
        'final_acc': fold_results['final_acc']
    })
    # 追加每类精度列
    for cls in class_names:
        summary_df[f'{cls}_precision'] = [fold_detail['final_class_precision'][cls] for fold_detail in
                                          fold_results['fold_details']]
    summary_df.to_csv(osp.join(fold_dir, "summary.csv"), index=False)
    return fold_results


if __name__ == '__main__':
    data_transforms = get_torch_transforms(img_size=params["img_size"])
    fold_results = train_k_fold(params, data_transforms, num_folds=5)
