#!/usr/bin/env python
"""
Faster R-CNN 验证脚本 - 使用YOLO验证指标体系
"""

import os
import json
import argparse
from pathlib import Path

import torch
import transforms
from tqdm import tqdm

# 导入自定义模块
from train_utils.YOLO_val import YOLOEvaluator
from backbone import resnet50_fpn_backbone
from network_files import FasterRCNN
from my_dataset import ClusterDataSet

# 添加这个导入以解决Windows下的问题
import sys

sys.path.append('.')


def create_model(num_classes, weights_path=None, device='cuda'):
    """
    创建Faster R-CNN模型并加载权重

    Args:
        num_classes: 类别数量（不包括背景）
        weights_path: 权重文件路径
        device: 设备

    Returns:
        模型实例
    """
    # 创建模型
    backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d)
    model = FasterRCNN(backbone=backbone, num_classes=num_classes + 1)  # +1 for background
    if weights_path:
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
        print(f"Loading weights from {weights_path}")
        weights_dict = torch.load(weights_path, map_location='cpu')
        # 处理权重字典格式
        if "model" in weights_dict:
            weights_dict = weights_dict["model"]
        # 加载权重
        missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
        if missing_keys:
            print(f"Missing keys in state_dict: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys in state_dict: {unexpected_keys}")
    return model


def create_data_loader(data_path, split_file, batch_size=1):
    """
    创建数据加载器
    Args:
        data_path: 数据路径
        split_file: 划分文件（如val.txt）
        batch_size: 批次大小
    Returns:
        数据加载器
    """
    # 数据转换

    data_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    # 创建数据集
    dataset = ClusterDataSet(data_path, data_transform, split_file)

    if os.name == 'nt':  # Windows系统
        num_workers = 0  # Windows下建议设为0避免共享内存问题
        multiprocessing_context = None
    else:  # Linux/Mac系统
        num_workers = min(4, os.cpu_count())  # 限制worker数量
        multiprocessing_context = 'spawn'

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=num_workers>0,  # Windows下禁用
        collate_fn=dataset.collate_fn,
        multiprocessing_context=multiprocessing_context
    )



def get_class_names(label_json_path):
    """
    从JSON文件获取类别名称
    """
    if not os.path.exists(label_json_path):
        print(f"Warning: Class label file not found: {label_json_path}")
        return {}
    with open(label_json_path, 'r') as f:
        class_dict = json.load(f)
    # 反转字典：id -> name
    names = {v: k for k, v in class_dict.items()}

    return names


def main(args):
    """
    主函数
    Args:
        args: 命令行参数
    """
    # 设备设置
    device = torch.device(args.device if torch.cuda.is_available() and args.device != 'cpu' else 'cpu')
    print(f"Using {device.type} device for evaluation")
    # 获取类别名称
    names = get_class_names(args.label_json)
    # 创建数据加载器
    print(f"Loading validation dataset from {args.data_path}")
    val_loader = create_data_loader(
        args.data_path,
        "val.txt",     # TODO
        batch_size=args.batch_size
    )


    # 创建模型
    model = create_model(
        num_classes=args.num_classes,
        weights_path=args.weights_path,
        device=device)

    # 创建验证器
    validator = YOLOEvaluator(
        model=model,
        dataloader=val_loader,
        save_dir=args.save_dir,
        device=device.type,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        max_det=args.max_det,
        plots=args.plots,
        save_json=args.save_json,
        save_txt=args.save_txt,
        half=args.half,
        names=names
    )
    # 执行评估
    print("\nStarting evaluation with YOLO metrics...")
    results = validator.evaluate()
    # 保存完整结果
    results_file = Path(args.save_dir) / 'final_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to: {args.save_dir}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Faster R-CNN evaluation with YOLO metrics",)
    # 基本参数
    parser.add_argument('--device', default='cuda', help='device (cuda, cpu, mps)')
    parser.add_argument('--num-classes', type=int, default=4, help='number of classes')
    parser.add_argument('--data-path', default=r'./ClusterDataset', help='dataset root path')
    parser.add_argument('--weights-path', default='./save_weight/best_model.pth',
                        type=str, help='model weights path')
    parser.add_argument('--label-json', default='./cluster_datasets_classes.json',
                        help='class labels JSON file')
    # 验证参数
    parser.add_argument('--batch-size', default=1, type=int, help='validation batch size')
    parser.add_argument('--conf-thres', type=float, default=0.77, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.7, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=300, help='maximum detections per image')
    # 输出参数
    parser.add_argument('--save-dir', default='./val_conf0.77', help='directory to save results')
    parser.add_argument('--plots', default=True, help='save plots')
    parser.add_argument('--save-json', default=True, help='save JSON results')
    parser.add_argument('--save-txt', default=True, help='save TXT results')
    parser.add_argument('--half', default=True, help='use half precision (FP16)')
    args = parser.parse_args()
    # 检查数据路径
    if not os.path.exists(args.data_path):
        # 查看当前工作目录

        print(f"Error: Data path '{args.data_path}' does not exist")
        exit(1)
    # 运行主函数
    main(args)