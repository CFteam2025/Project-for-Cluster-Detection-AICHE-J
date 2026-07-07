import math
import sys
import time
import numpy as np  # 新增：用于计算逐类AP
import torch

from .coco_utils import get_coco_api_from_dataset
from .coco_eval import CocoEvaluator
import train_utils.distributed_utils as utils


def train_one_epoch(model, optimizer, data_loader, device, epoch,
                    print_freq=50, warmup=False, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0 and warmup is True:  # 当训练第一轮（epoch=0）时，启用warmup训练方式，可理解为热身训练
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    mloss = torch.zeros(1).to(device)  # mean losses
    for i, [images, targets] in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # 混合精度训练上下文管理器，如果在CPU环境中不起任何作用
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purpose
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()
        # 记录训练损失
        mloss = (mloss * i + loss_value) / (i + 1)  # update mean losses

        if not math.isfinite(loss_value):  # 当计算的损失为无穷大时停止训练
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:  # 第一轮使用warmup训练方式
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        now_lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=now_lr)

    return mloss, now_lr


@torch.no_grad()
def evaluate(model, data_loader, device, num_classes=None, class_names=None):  # 新增参数：类别数、类别名
    """
    扩展evaluate函数：返回整体coco_info + 逐类AP值
    :param model: 模型
    :param data_loader: 验证集dataloader
    :param device: 设备
    :param num_classes: 总类别数（含背景）
    :param class_names: 类别名称列表（不含背景）
    :return: coco_info（整体指标）、per_class_ap（逐类AP字典）
    """
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Val: "

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for image, targets in metric_logger.log_every(data_loader, 100, header):
        image = list(img.to(device) for img in image)

        # 当使用CPU时，跳过GPU相关指令
        if device != torch.device("cpu"):
            torch.cuda.synchronize(device)

        model_time = time.time()
        outputs = model(image)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}

        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    # 原有逻辑：提取整体coco指标
    coco_info = coco_evaluator.coco_eval[iou_types[0]].stats.tolist()  # numpy to list

    # ===================== 新增核心逻辑：提取逐类AP =====================
    per_class_ap = {}
    # 仅当类别数有效且coco_eval有评估结果时，计算逐类AP
    if num_classes is not None and hasattr(coco_evaluator.coco_eval[iou_types[0]], "eval"):
        coco_eval = coco_evaluator.coco_eval[iou_types[0]]
        # precision矩阵 shape: [IoU阈值数(10), 召回率数(101), 类别数, 面积数(4), 最大检测数(3)]
        # IoU阈值：0.5:0.95（步长0.05）、召回率：0~1（步长0.01）、面积：all/small/medium/large、最大检测数：1/10/100
        precision = coco_eval.eval["precision"]

        # 过滤条件：IoU=0.5:0.95（全部）、面积=all（索引0）、最大检测数=100（索引2）
        precision = precision[:, :, :, 0, 2]  # 简化为 [10, 101, 类别数]

        # 遍历每个类别计算AP（背景类为0，跳过）
        for cls_idx in range(precision.shape[2]):
            # 取出该类的精度值，忽略nan（无预测的情况）
            cls_precision = precision[:, :, cls_idx]
            cls_precision = cls_precision[~np.isnan(cls_precision)]

            # 计算该类AP：对召回率取平均（COCO标准），转百分比并保留4位小数
            if len(cls_precision) == 0:
                ap = 0.0
            else:
                ap = round(np.mean(cls_precision) * 100, 4)  # 转百分比，保留4位

            # 确定类别名（优先用传入的class_names，否则用默认命名）
            if class_names is not None and cls_idx < len(class_names):
                cls_name = class_names[cls_idx]
            else:
                cls_name = f"class_{cls_idx + 1}"  # 类别ID从1开始（背景为0）

            per_class_ap[cls_name] = ap

    # 返回整体指标 + 逐类AP
    return coco_info, per_class_ap


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    return iou_types