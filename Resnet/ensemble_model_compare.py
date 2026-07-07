import torch
import torch.nn.functional as F
import timm
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.data.build import build_dataloader, build_yolo_dataset
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils.torch_utils import de_parallel
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils.ops import Profile
from ultralytics.utils import DEFAULT_CFG, LOGGER, TQDM, colorstr, callbacks
import copy
import os.path as osp
from pathlib import Path
import matplotlib.pyplot as plt
import random
import numpy as np
from torchvision import transforms


class ClassifierTransforms:
    def __init__(self, img_size=224):
        # 训练/验证时的预处理（必须与分类模型训练代码完全一致）
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),  # 与训练对齐
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 与训练对齐
        ])
        self.val_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),  # 与训练对齐（若训练验证集用Resize）
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 与训练对齐
        ])

    def __call__(self, crop, is_train=False):
        """
        分类模型的预处理（与训练严格对齐）
        :param crop: [C, H, W] 的 Tensor（GPU/CPU）
        :param is_train: 是否用训练时的增强（路线二验证时一般设为False）
        :return: 预处理后的 Tensor
        """
        # 转换为PIL（若训练时用PIL处理）
        crop_pil = transforms.ToPILImage()(crop.cpu())  # 若已在GPU，先转CPU
        if is_train:
            return self.train_transform(crop_pil).to(crop.device)  # 转回原设备
        else:
            return self.val_transform(crop_pil).to(crop.device)


# ======================== 预处理流程结束 ========================


class EnsembleValidator(DetectionValidator):
    def __init__(self, cfg, det_model_path, cls_model_path=None, num_classes=4):
        super().__init__(args=cfg, save_dir=cfg.save_dir)
        self.det_model_path = det_model_path
        self.cls_model_path = cls_model_path
        self.num_classes = num_classes
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        callbacks.add_integration_callbacks(self)
        self.load_models()
        self.build_dataset()
        # 初始化分类模型的预处理（与训练对齐）
        self.cls_transforms = ClassifierTransforms(img_size=224)  # 与训练的img_size对齐

    def load_models(self):
        # 加载检测模型 (两条路线共享)
        self.det_model = AutoBackend(
            weights=self.det_model_path,
            device=self.device,
            fp16=self.args.half,
            verbose=True
        )
        self.stride = self.det_model.stride
        imgsz = check_imgsz(self.args.imgsz, stride=self.stride)
        self.args.batch = 1
        LOGGER.info(f"Forcing batch=1 square inference (1,3,{imgsz},{imgsz}) ")
        self.det_model.eval()
        self.det_model.warmup(imgsz=(1, 3, imgsz, imgsz))
        LOGGER.info(f"检测模型加载完成: {self.det_model_path}")

        # 仅路线2需要加载分类器
        if self.cls_model_path:
            self.det_model.names = {0: 'U-type', 1: "inverted-U", 2: "chain", 3: "special"}
            # 初始化分类模型（与训练代码对齐）
            model = timm.create_model("resnet50d", pretrained=False)
            model.fc = torch.nn.Linear(model.fc.in_features, self.num_classes)
            self.classifier = WrappedModel(model)
            state_dict = torch.load(self.cls_model_path, map_location=self.device)
            self.classifier.load_state_dict(state_dict)
            self.classifier.to(self.device).eval()
            LOGGER.info(f"分类模型加载完成: {self.cls_model_path}")

    def build_dataset(self):
        self.data = check_det_dataset(self.args.data)
        self.args.workers = 0
        self.dataset = build_yolo_dataset(
            self.args,
            img_path=self.data.get(self.args.split),
            batch=self.args.batch,
            data=self.args.data,
            mode='val',
            rect=self.args.rect,
            stride=int(torch.tensor(self.det_model.stride).max()))
        assert self.dataset is not None, "数据集初始化失败，请检查参数是否正确"
        self.dataloader = build_dataloader(
            self.dataset,
            batch=self.args.batch,
            workers=self.args.workers,
            shuffle=False,
            rank=-1)

    def validate_route(self):
        self.run_callbacks("on_val_start")
        dt = (Profile(device=self.device), Profile(device=self.device),
              Profile(device=self.device), Profile(device=self.device),)
        bar = TQDM(self.dataloader, desc=self.get_desc(), total=len(self.dataloader))
        self.init_metrics(de_parallel(self.det_model))
        self.jdict = []
        for batch_i, batch in enumerate(bar):
            self.run_callbacks("on_val_batch_start")
            self.batch_i = batch_i
            # Preprocess
            with dt[0]:
                batch = self.preprocess(batch)
            # Inference
            with dt[1]:
                preds = self.det_model(batch["img"], augment=self.args.augment)
            # Loss
            with dt[2]:
                if self.training:
                    pass
            # postprocess
            with dt[3]:
                preds = self.postprocess(preds)
                if self.cls_model_path is not None:
                    new_preds = []
                    orig_images = batch['img']
                    pred_device = preds[0].device if preds else self.device
                    for img_idx, pred in enumerate(preds):
                        img = orig_images[img_idx]
                        img_preds = []
                        for det in pred:
                            x1, y1, x2, y2, conf, cls = det.unbind(-1)
                            xi1, yi1, xi2, yi2 = int(x1), int(y1), int(x2), int(y2)
                            # 裁剪目标区域（GPU Tensor）
                            crop = img[:, yi1:yi2, xi1:xi2]
                            if crop.numel() == 0:
                                continue
                            # 与分类模型训练时的预处理严格对齐（禁用训练增强）
                            crop_normalized = self.cls_transforms(crop, is_train=False)
                            with torch.no_grad():
                                cls_logits = self.classifier(crop_normalized.unsqueeze(0))  # 加batch维度
                                cls_pred = torch.argmax(cls_logits, dim=1).item()
                            if cls_pred == 1:  # 原预测为inverted-U，修正为chain
                                cls_pred = 2
                            elif cls_pred == 2:  # 原预测为chain，修正为inverted-U
                                cls_pred = 1
                            img_preds.append([x1, y1, x2, y2, conf, cls_pred])
                        # 处理当前图片的所有检测框
                        if img_preds:
                            new_pred = torch.tensor(img_preds, dtype=torch.float32, device=pred_device)
                        else:
                            new_pred = torch.zeros((0, 6), dtype=torch.float32, device=pred_device)
                        new_preds.append(new_pred)
                    preds = new_preds

            self.update_metrics(preds, batch)
            if self.args.plots:
                self.plot_val_samples(batch, batch_i)
                self.plot_predictions(batch, preds, batch_i)
            self.run_callbacks("on_val_batch_end")
        stats = self.get_stats()
        self.check_stats(stats)
        self.speed = dict(zip(self.speed.keys(), (x.t / len(self.dataloader.dataset) * 1e3 for x in dt)))
        self.finalize_metrics()
        self.print_results()
        data = {
            "类别名称": self.det_model.names,
            "精度(Precision)": np.array(self.metrics.box.p).tolist(),
            "召回率(Recall)": np.array(self.metrics.box.r).tolist(),
            "F1分数": np.array(self.metrics.box.f1).tolist(),
            "P曲线": np.array(self.metrics.box.p_curve).tolist(),
            "R曲线": np.array(self.metrics.box.r_curve).tolist(),
            "AP50-95": np.array(self.metrics.box.all_ap).tolist(),
            "类别索引": np.array(self.metrics.box.ap_class_index).tolist()
        }
        with open('eval_results.txt', 'a', encoding='utf-8') as f:
            f.write("\n===== YOLO单目标检测+Resnet50 =====\n") if self.cls_model_path else f.write(
                "\n===== YOLO四目标检测 =====\n")
            for key, value in data.items():
                f.write(f"{key}: {value}\n")
            for key, value in stats.items():
                f.write(f"{key}: {value}\n")

        self.run_callbacks("on_val_end")
        LOGGER.info("Speed: %.1fms preprocess, %.1fms inference, %.1fms loss, %.1fms postprocess per image"
                    % tuple(self.speed.values()))
        if self.args.plots or self.args.save_json:
            LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}")
        print("\n--------------------------------------------------------------")
        return stats


class WrappedModel(torch.nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.model = original_model

    def forward(self, x):
        return self.model(x)


def compare_routes(det_model_route1, det_model_route2, cls_model, data_cfg):
    cfg = copy.deepcopy(DEFAULT_CFG)
    cfg.data = data_cfg
    cfg.conf = 0.001
    cfg.iou = 0.7
    cfg.split = 'val'   # todo
    save_dir = "./runs/val_conf0.001"
    cfg.device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    cfg.save_dir = Path(osp.join(save_dir, "model1"))
    validator_route1 = EnsembleValidator(cfg, det_model_path=det_model_route1)
    results_route1 = validator_route1.validate_route()

    cfg.save_dir = Path(osp.join(save_dir, "model2"))
    validator_route2 = EnsembleValidator(cfg,
                                         det_model_path=det_model_route2,
                                         cls_model_path=cls_model,
                                         num_classes=4)
    results_route2 = validator_route2.validate_route()
    return results_route1, results_route2


if __name__ == "__main__":
    DET_MODEL_ROUTE1 = r"D:\project_SJ\CUP-cluster-detection\utralytics\scripts\runs\1.base_model\weights\best.pt"
    DET_MODEL_ROUTE2 = r"D:\project_SJ\CUP-cluster-detection\utralytics\scripts\runs\oneclass_model2\weights\best.pt"
    CLS_MODEL = r"D:\project_SJ\CUP-cluster-detection\utralytics\scripts\best_model_fold3.pth"
    DATA_CONFIG = r"D:\project_SJ\ultralytics-main\datasets\dataset-v10-four-kfold\fold_4\dataset_fold_4.yaml"
    results_route1, results_route2 = compare_routes(det_model_route1=DET_MODEL_ROUTE1,
                                                    det_model_route2=DET_MODEL_ROUTE2,
                                                    cls_model=CLS_MODEL,
                                                    data_cfg=DATA_CONFIG)
    print("路线1的结果为：", results_route1)
    print("路线2的结果为：", results_route2)
