# Ultralytics YOLO 风格验证器，适配Faster R-CNN
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from train_utils.metrics import DetMetrics, ConfusionMatrix, ap_per_class, box_iou


class YOLOEvaluator:
    """
    Faster R-CNN验证器，使用YOLO验证指标体系
    """

    def __init__(self, model, dataloader, save_dir='./runs/val', device='cuda',
                 conf_thres=0.001, iou_thres=0.6, max_det=300, plots=True,
                 save_json=False, save_txt=False, half=False, names=None):
        """
        初始化验证器

        Args:
            model: Faster R-CNN模型
            dataloader: 验证数据加载器
            save_dir: 结果保存目录
            device: 设备
            conf_thres: 置信度阈值
            iou_thres: NMS的IoU阈值
            max_det: 每张图最大检测数
            plots: 是否绘制图表
            save_json: 是否保存JSON结果
            save_txt: 是否保存TXT结果
            half: 是否使用半精度
            names: 类别名称列表
        """
        self.model = model
        self.dataloader = dataloader
        self.save_dir = Path(save_dir)
        self.device = torch.device(device)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.plots = plots
        self.save_json = save_json
        self.save_txt = save_txt
        self.half = half and self.device.type != 'cpu'

        # 创建保存目录
        self.save_dir.mkdir(parents=True, exist_ok=True)
        if self.save_txt:
            (self.save_dir / 'labels').mkdir(parents=True, exist_ok=True)

        # 类别信息
        self.names = names if names is not None else {i: f'class{i}' for i in range(91)}
        self.nc = len(self.names)

        # 初始化指标
        self.iouv = torch.linspace(0.5, 0.95, 10)  # IoU vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()

        # 统计信息
        self.stats = {
            'tp': [],  # True positives
            'conf': [],  # Confidence scores
            'pred_cls': [],  # Predicted classes
            'target_cls': [],  # Target classes
            'target_img': []  # Target image indices
        }
        self.seen = 0
        self.jdict = []  # 用于保存JSON结果

        # 初始化Metrics
        self.metrics = DetMetrics(save_dir=self.save_dir, plot=self.plots, names=self.names)
        self.confusion_matrix = ConfusionMatrix(nc=self.nc, conf=self.conf_thres)

        # 速度统计
        self.speed = {'preprocess': 0.0, 'inference': 0.0, 'loss': 0.0, 'postprocess': 0.0}

        # 将模型移动到设备
        self.model.to(self.device)
        if self.half:
            self.model.half()
        self.model.eval()

    def preprocess(self, batch):
        """
        预处理批次数据

        Args:
            batch: (images, targets) 或 images

        Returns:
            预处理后的数据
        """
        images, targets = batch

        # 确保images是列表
        if isinstance(images, list):
            images = [img.to(self.device, non_blocking=True) for img in images]
            # 创建批次张量（Faster R-CNN通常需要列表格式）
            processed_images = images
        else:
            images = images.to(self.device, non_blocking=True)
            processed_images = images

        # 转换为半精度
        if self.half:
            if isinstance(processed_images, list):
                processed_images = [img.half() for img in processed_images]
            else:
                processed_images = processed_images.half()

        return {
            'img': processed_images,
            'targets': targets,
            'ori_shapes': [(img.shape[1], img.shape[2]) for img in processed_images] if processed_images else [],
            'img_files': [target['image_id'] for target in targets] if targets else []
        }

    def postprocess(self, predictions, targets, orig_shapes):
        """
        后处理Faster R-CNN的输出

        Args:
            predictions: Faster R-CNN的输出列表
            targets: 真实标注
            orig_shapes: 原始图像尺寸

        Returns:
            格式化的检测结果
        """
        all_detections = []

        for i, pred in enumerate(predictions):
            if pred is None or len(pred['boxes']) == 0:
                all_detections.append(torch.empty((0, 6), device=self.device))
                continue

            # 获取预测框、分数和类别
            boxes = pred['boxes']
            scores = pred['scores']
            labels = pred['labels']

            # 应用置信度阈值
            keep = scores >= self.conf_thres
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]

            # 应用NMS
            if len(boxes) > 0:
                keep = torch.ops.torchvision.nms(boxes, scores, self.iou_thres)
                keep = keep[:self.max_det]
                boxes = boxes[keep]
                scores = scores[keep]
                labels = labels[keep]

            # 格式化为[x1, y1, x2, y2, conf, cls]
            if len(boxes) > 0:
                detections = torch.cat([
                    boxes,
                    scores.unsqueeze(1),
                    labels.float().unsqueeze(1)
                ], dim=1)
            else:
                detections = torch.empty((0, 6), device=self.device)

            all_detections.append(detections)

        return all_detections

    def prepare_targets(self, targets):
        """
        准备真实标注数据

        Args:
            targets: 原始标注

        Returns:
            格式化的真实标注
        """
        formatted_targets = []

        for i, target in enumerate(targets):
            if 'boxes' not in target or len(target['boxes']) == 0:
                formatted_targets.append({
                    'boxes': torch.empty((0, 4), device=self.device),
                    'labels': torch.empty((0,), device=self.device, dtype=torch.int64)
                })
                continue

            boxes = target['boxes'].to(self.device)
            labels = target['labels'].to(self.device)

            formatted_targets.append({
                'boxes': boxes,
                'labels': labels
            })

        return formatted_targets

    def update_metrics(self, preds, targets, batch_idx):
        """
        更新评估指标

        Args:
            preds: 预测结果列表
            targets: 真实标注列表
            batch_idx: 批次索引
        """
        for i, (pred, target) in enumerate(zip(preds, targets)):
            self.seen += 1

            # 获取预测和真实数据
            npr = len(pred)  # 预测数量
            nl = len(target['labels']) if target is not None and len(target['labels']) > 0 else 0  # 真实数量

            # 初始化统计
            stat = {
                'conf': torch.zeros(0, device=self.device),
                'pred_cls': torch.zeros(0, device=self.device),
                'tp': torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
                'target_cls': torch.empty(0, device=self.device, dtype=torch.int64),
                'target_img': torch.empty(0, device=self.device, dtype=torch.int64)
            }

            if nl > 0:
                stat['target_cls'] = target['labels']
                stat['target_img'] = torch.tensor([batch_idx * len(preds) + i], device=self.device)

            # 如果没有预测
            if npr == 0:
                if nl > 0:
                    for k in self.stats.keys():
                        self.stats[k].append(stat[k])
                    if self.plots:
                        self.confusion_matrix.process_batch(
                            detections=None,
                            gt_bboxes=target['boxes'],
                            gt_cls=target['labels']
                        )
                continue

            # 准备预测数据
            stat['conf'] = pred[:, 4]
            stat['pred_cls'] = pred[:, 5].int()

            # 如果有真实标注，计算匹配
            if nl > 0:
                # 计算IoU
                iou_matrix = box_iou(target['boxes'], pred[:, :4])

                # 使用YOLO的匹配方法
                stat['tp'] = self.match_predictions(
                    pred[:, 5].int(),
                    target['labels'],
                    iou_matrix
                )

                # 更新混淆矩阵
                if self.plots:
                    self.confusion_matrix.process_batch(
                        detections=pred,
                        gt_bboxes=target['boxes'],
                        gt_cls=target['labels']
                    )

            # 保存统计
            for k in self.stats.keys():
                self.stats[k].append(stat[k])

    def match_predictions(self, pred_classes, true_classes, iou, use_scipy=False):
        """
        匹配预测和真实标注（从YOLO的BaseValidator复制）
        """
        # Dx10 matrix, where D - detections, 10 - IoU thresholds
        correct = np.zeros((pred_classes.shape[0], self.iouv.shape[0])).astype(bool)

        # 如果没有真实标注，返回全False
        if len(true_classes) == 0:
            return torch.tensor(correct, dtype=torch.bool, device=pred_classes.device)

        # LxD matrix where L - labels (rows), D - detections (columns)
        correct_class = true_classes[:, None] == pred_classes
        iou = iou * correct_class  # zero out the wrong classes
        iou = iou.cpu().numpy()

        for i, threshold in enumerate(self.iouv.cpu().tolist()):
            matches = np.nonzero(iou >= threshold)  # IoU > threshold and classes match
            matches = np.array(matches).T

            if matches.shape[0]:
                if matches.shape[0] > 1:
                    matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                correct[matches[:, 1].astype(int), i] = True

        return torch.tensor(correct, dtype=torch.bool, device=pred_classes.device)

    def get_stats(self):
        """
        获取评估统计结果
        """
        # 合并所有统计
        stats = {}
        for k, v in self.stats.items():
            if k == 'target_img':
                continue
            if len(v) > 0:
                stats[k] = torch.cat(v, 0).cpu().numpy()
            else:
                stats[k] = np.array([])

        # 如果没有统计结果，返回空指标
        if len(stats) == 0 or len(stats['tp']) == 0:
            return self.metrics.results_dict

        # 更新Metrics
        self.metrics.process(
            stats['tp'],
            stats['conf'],
            stats['pred_cls'],
            stats['target_cls']
        )

        return self.metrics.results_dict

    def print_results(self):
        """
        打印评估结果
        """
        results = self.metrics.results_dict

        # 打印标题
        LOGGER.info(("%22s" + "%11s" * 6) % ("Class", "Images", "Instances", "P", "R", "mAP50", "mAP50-95"))

        # 打印所有类别的平均结果
        if hasattr(self.metrics, 'box'):
            mp, mr, map50, map95 = self.metrics.box.mean_results()
            LOGGER.info(("%22s" + "%11i" * 2 + "%11.3g" * 4) %
                        ("all", self.seen, sum(self.metrics.box.ap_class_index), mp, mr, map50, map95))

        # 打印每个类别的结果
        if self.nc > 1 and len(self.metrics.box.ap_class_index) > 0:
            for i, c in enumerate(self.metrics.box.ap_class_index):
                p, r, ap50, ap = self.metrics.box.class_result(i)
                LOGGER.info(("%22s" + "%11i" * 2 + "%11.3g" * 4) %
                            (self.names.get(c, f'class{c}'), self.seen, 1, p, r, ap50, ap))

    def save_results(self):
        """
        保存评估结果
        """
        # 保存结果到TXT文件
        results_file = self.save_dir / 'results.txt'
        with open(results_file, 'w') as f:
            results = self.metrics.results_dict
            f.write("Faster R-CNN Evaluation Results\n")
            f.write("=" * 50 + "\n")
            f.write(f"Precision(B): {results.get('metrics/precision(B)', 0):.4f}\n")
            f.write(f"Recall(B): {results.get('metrics/recall(B)', 0):.4f}\n")
            f.write(f"mAP50(B): {results.get('metrics/mAP50(B)', 0):.4f}\n")
            f.write(f"mAP50-95(B): {results.get('metrics/mAP50-95(B)', 0):.4f}\n")
            f.write(f"Fitness: {results.get('fitness', 0):.4f}\n")

        # 保存混淆矩阵
        if self.plots:
            self.confusion_matrix.plot(
                save_dir=self.save_dir,
                names=self.names,
                normalize=True
            )
            self.confusion_matrix.plot(
                save_dir=self.save_dir,
                names=self.names,
                normalize=False
            )

    def evaluate(self):
        """
        执行完整的评估流程
        """
        print(f"Starting evaluation on {len(self.dataloader)} batches...")

        # 计时
        start_time = time.time()

        # 评估循环
        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.dataloader):
                # 预处理
                batch_start = time.time()
                processed_batch = self.preprocess(batch)
                preprocess_time = time.time() - batch_start

                # 推理
                inference_start = time.time()
                predictions = self.model(processed_batch['img'])
                inference_time = time.time() - inference_start

                # 后处理
                postprocess_start = time.time()
                formatted_targets = self.prepare_targets(processed_batch['targets'])
                detections = self.postprocess(
                    predictions,
                    formatted_targets,
                    processed_batch['ori_shapes']
                )
                postprocess_time = time.time() - postprocess_start

                # 更新指标
                self.update_metrics(detections, formatted_targets, batch_idx)

                # 更新速度统计
                self.speed['preprocess'] += preprocess_time
                self.speed['inference'] += inference_time
                self.speed['postprocess'] += postprocess_time

                # 打印进度
                if batch_idx % 10 == 0:
                    print(f"Processed {batch_idx}/{len(self.dataloader)} batches")

        # 计算最终指标
        total_time = time.time() - start_time
        results = self.get_stats()

        # 打印结果
        print("\n" + "=" * 60)
        print("Evaluation Results:")
        print("=" * 60)
        self.print_results()

        # 打印速度统计
        num_samples = len(self.dataloader.dataset)
        print("\nSpeed Statistics:")
        print(f"Total time: {total_time:.2f}s")
        print(f"Preprocess: {self.speed['preprocess'] / num_samples * 1000:.1f}ms per image")
        print(f"Inference: {self.speed['inference'] / num_samples * 1000:.1f}ms per image")
        print(f"Postprocess: {self.speed['postprocess'] / num_samples * 1000:.1f}ms per image")

        # 保存结果
        self.save_results()

        return results


# 简单的日志类
class LOGGER:
    @staticmethod
    def info(msg):
        print(msg)