import random
import math
import numpy as np
import cv2
from PIL import Image
import torch
from torchvision.transforms import functional as F


class Compose(object):
    """组合多个transform函数"""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target



class ToTensor(object):
    """将PIL图像转为Tensor，并添加灰度图处理"""

    def __call__(self, image, target):
        if isinstance(image, np.ndarray):
            # numpy数组转Tensor
            if len(image.shape) == 2:
                # 灰度图转为3通道
                image = np.stack([image] * 3, axis=-1)
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        elif isinstance(image, Image.Image):
            # PIL转Tensor
            image = F.to_tensor(image)
            # 如果是单通道，转为3通道
            if image.shape[0] == 1:
                image = image.repeat(3, 1, 1)
        return image, target


class RandomHorizontalFlip(object):
    """随机水平翻转图像以及bboxes"""

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            if isinstance(image, torch.Tensor):
                height, width = image.shape[-2:]
                image = image.flip(-1)
            elif isinstance(image, np.ndarray):
                height, width = image.shape[:2]
                image = image[:, ::-1, :]
            else:
                width, height = image.size
                image = image.transpose(Image.FLIP_LEFT_RIGHT)

            if "boxes" in target and len(target["boxes"]) > 0:
                bbox = target["boxes"]
                if isinstance(bbox, torch.Tensor):
                    bbox = bbox.clone()
                bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
                target["boxes"] = bbox
        return image, target


class LetterBox(object):
    """LetterBox填充和调整大小"""

    def __init__(self, new_shape=(640, 640), stride=32, auto=True, scaleFill=False, scaleup=True, center=True):
        self.new_shape = new_shape
        self.stride = stride
        self.auto = auto
        self.scaleFill = scaleFill
        self.scaleup = scaleup
        self.center = center

    def __call__(self, image, target):
        # 确保图像是numpy格式，并正确处理数据类型
        if isinstance(image, torch.Tensor):
            # Tensor转numpy，检查值范围
            img_np = image.permute(1, 2, 0).numpy()

            # 判断Tensor的值范围
            if img_np.max() <= 1.0 and img_np.min() >= 0:
                # 范围是[0,1]，需要转[0,255]
                img_np = (img_np * 255.0).astype(np.uint8)
            elif img_np.max() <= 255 and img_np.min() >= 0:
                # 范围已经是[0,255]
                img_np = img_np.astype(np.uint8)
            else:
                # 其他情况，直接转为uint8
                img_np = img_np.astype(np.uint8)

            is_tensor = True
        elif isinstance(image, Image.Image):
            img_np = np.array(image)
            is_tensor = False
        else:
            img_np = image
            is_tensor = False

        # 确保是3通道，值范围[0,255]
        if len(img_np.shape) == 2:
            img_np = np.stack([img_np] * 3, axis=-1)
        elif img_np.shape[2] == 1:
            img_np = np.repeat(img_np, 3, axis=2)
        elif img_np.shape[2] == 4:
            img_np = img_np[:, :, :3]

        # 确保数据类型是uint8
        if img_np.dtype != np.uint8:
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
            else:
                img_np = img_np.astype(np.uint8)

        h, w = img_np.shape[:2]
        new_h, new_w = self.new_shape

        # 缩放比例
        r = min(new_h / h, new_w / w)
        if not self.scaleup:
            r = min(r, 1.0)

        # 计算新的尺寸
        new_unpad = int(round(w * r)), int(round(h * r))
        dw, dh = new_w - new_unpad[0], new_h - new_unpad[1]

        if self.auto:
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)
        elif self.scaleFill:
            dw, dh = 0.0, 0.0
            new_unpad = (new_w, new_h)
            r = new_w / w  # 修复：保持为scalar

        if self.center:
            dw /= 2.0
            dh /= 2.0

        # 调整图像大小
        if new_unpad != (w, h):
            img_np = cv2.resize(img_np, new_unpad, interpolation=cv2.INTER_LINEAR)

        # 添加填充
        top, bottom = int(round(dh - 0.1)) if self.center else 0, int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)) if self.center else 0, int(round(dw + 0.1))

        # 确保top/bottom/left/right是非负数
        top = max(0, top)
        bottom = max(0, bottom)
        left = max(0, left)
        right = max(0, right)

        # 创建新的图像并填充
        img_new = np.full((new_h, new_w, 3), 114, dtype=np.uint8)

        # 计算放置位置
        y_start = top
        y_end = top + new_unpad[1]
        x_start = left
        x_end = left + new_unpad[0]

        # 确保不越界
        y_end = min(y_end, new_h)
        x_end = min(x_end, new_w)
        img_resized = img_np[:y_end - y_start, :x_end - x_start]

        img_new[y_start:y_end, x_start:x_end] = img_resized

        # 更新所有目标字段
        if "boxes" in target and len(target["boxes"]) > 0:
            boxes = target["boxes"]
            if isinstance(boxes, torch.Tensor):
                boxes = boxes.numpy().astype(np.float32)

            # 缩放并平移边界框
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * r + left
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * r + top

            # 确保边界框在图像范围内
            boxes[:, 0] = np.clip(boxes[:, 0], 0, new_w)
            boxes[:, 1] = np.clip(boxes[:, 1], 0, new_h)
            boxes[:, 2] = np.clip(boxes[:, 2], 0, new_w)
            boxes[:, 3] = np.clip(boxes[:, 3], 0, new_h)

            target["boxes"] = torch.from_numpy(boxes)

            # 更新面积
            if "area" in target:
                area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                target["area"] = torch.from_numpy(area)

        # 转换图像格式
        if is_tensor:
            # 保持为numpy，由ToTensor处理
            image = img_new
        else:
            image = Image.fromarray(img_new.astype(np.uint8))

        return image, target


class RandomHSV(object):
    """随机调整HSV"""

    def __init__(self, hgain=0.015, sgain=0.7, vgain=0.4):
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain

    def __call__(self, image, target):
        if self.hgain or self.sgain or self.vgain:
            # 转换图像格式
            if isinstance(image, torch.Tensor):
                img_np = image.permute(1, 2, 0).numpy() * 255.0
                img_np = img_np.astype(np.uint8)
                is_tensor = True
            elif isinstance(image, Image.Image):
                img_np = np.array(image)
                is_tensor = False
            else:
                img_np = image
                is_tensor = False

            # 确保是3通道BGR格式
            if len(img_np.shape) == 2:
                img_np = np.stack([img_np] * 3, axis=-1)
            elif img_np.shape[2] == 1:
                img_np = np.repeat(img_np, 3, axis=2)
            elif img_np.shape[2] == 4:
                img_np = img_np[:, :, :3]

            # 已经是RGB，转为BGR用于OpenCV
            img_bgr = img_np[:, :, ::-1]

            # HSV增强
            r = np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain, self.vgain] + 1

            hue, sat, val = cv2.split(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV))
            dtype = img_bgr.dtype

            x = np.arange(0, 256, dtype=r.dtype)
            lut_hue = ((x * r[0]) % 180).astype(dtype)
            lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
            lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

            im_hsv = cv2.merge((cv2.LUT(hue, lut_hue),
                                cv2.LUT(sat, lut_sat),
                                cv2.LUT(val, lut_val)))
            img_bgr = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR)

            # BGR转回RGB
            img_np = img_bgr[:, :, ::-1]

            # 转换回原始格式
            if is_tensor:
                image = img_np  # 保持numpy格式
            else:
                image = Image.fromarray(img_np.astype(np.uint8))

        return image, target


class RandomFlip(object):
    """随机翻转"""

    def __init__(self, p=0.5, direction="horizontal"):
        self.p = p
        self.direction = direction

    def __call__(self, image, target):
        if random.random() < self.p:
            if isinstance(image, torch.Tensor):
                if self.direction == "horizontal":
                    image = image.flip(-1)
                    w = image.shape[-1]
                else:
                    image = image.flip(-2)
                    h = image.shape[-2]
            elif isinstance(image, np.ndarray):
                h, w = image.shape[:2]
                if self.direction == "horizontal":
                    image = image[:, ::-1, :]
                else:
                    image = image[::-1, :, :]
            else:
                w, h = image.size
                if self.direction == "horizontal":
                    image = image.transpose(Image.FLIP_LEFT_RIGHT)
                else:
                    image = image.transpose(Image.FLIP_TOP_BOTTOM)

            if "boxes" in target and len(target["boxes"]) > 0:
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    boxes = boxes.clone()
                if self.direction == "horizontal":
                    boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
                else:
                    boxes[:, [1, 3]] = h - boxes[:, [3, 1]]
                target["boxes"] = boxes

        return image, target


class RandomPerspective(object):
    """随机透视和仿射变换"""

    def __init__(self, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, border=(0, 0)):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.border = border

    def apply_bboxes(self, bboxes, M, img_w, img_h, perspective=True):
        """
        将仿射/透视变换应用到bboxes

        Args:
            bboxes: (n, 4) 格式为[x1, y1, x2, y2]
            M: 3x3 变换矩阵
            img_w: 图像宽度
            img_h: 图像高度
            perspective: 是否为透视变换
        """
        n = len(bboxes)
        if n == 0:
            return bboxes

        # 为每个bbox的4个角点添加齐次坐标
        xy = np.ones((n * 4, 3), dtype=np.float32)
        # x1y1, x2y2, x1y2, x2y1
        xy[:, :2] = bboxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)

        # 应用变换
        xy = xy @ M.T

        # 如果是透视变换，需要进行透视除法
        if perspective:
            xy = xy[:, :2] / xy[:, 2:3]
        else:
            xy = xy[:, :2]

        # 重塑为 (n, 8)
        xy = xy.reshape(n, 8)

        # 从变换后的角点创建新的bbox
        x = xy[:, [0, 2, 4, 6]]  # x坐标
        y = xy[:, [1, 3, 5, 7]]  # y坐标

        # 计算新的bbox: [x_min, y_min, x_max, y_max]
        new_bboxes = np.concatenate((x.min(1, keepdims=True),
                                     y.min(1, keepdims=True),
                                     x.max(1, keepdims=True),
                                     y.max(1, keepdims=True)), axis=1)

        # 裁剪到图像边界
        new_bboxes[:, [0, 2]] = new_bboxes[:, [0, 2]].clip(0, img_w)
        new_bboxes[:, [1, 3]] = new_bboxes[:, [1, 3]].clip(0, img_h)

        return new_bboxes

    def box_candidates(self, box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1, eps=1e-16):
        """
        过滤掉变换后不符合条件的bbox

        Args:
            box1: 变换前的bbox [n, 4]
            box2: 变换后的bbox [n, 4]
        """
        w1 = box1[:, 2] - box1[:, 0]
        h1 = box1[:, 3] - box1[:, 1]
        w2 = box2[:, 2] - box2[:, 0]
        h2 = box2[:, 3] - box2[:, 1]

        # 宽高比
        ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))

        # 判断条件
        candidates = ((w2 > wh_thr) &
                      (h2 > wh_thr) &
                      (w2 * h2 / (w1 * h1 + eps) > area_thr) &
                      (ar < ar_thr))

        return candidates

    def __call__(self, image, target):
        """应用随机透视变换"""
        # 如果没有变换参数，直接返回
        if (self.degrees == 0 and self.translate == 0 and
                self.scale == 0 and self.shear == 0 and abs(self.perspective) < 1e-6):
            return image, target

        # 转换图像格式
        if isinstance(image, torch.Tensor):
            img_np = image.permute(1, 2, 0).numpy()
            h, w = img_np.shape[:2]
            is_tensor = True
        elif isinstance(image, Image.Image):
            img_np = np.array(image)
            h, w = img_np.shape[:2]
            is_tensor = False
        else:
            img_np = image
            h, w = img_np.shape[:2]
            is_tensor = False

        # 添加边框（如果有）
        if self.border[0] > 0 or self.border[1] > 0:
            top = bottom = self.border[0]
            left = right = self.border[1]
            img_np = cv2.copyMakeBorder(img_np, top, bottom, left, right,
                                        cv2.BORDER_CONSTANT, value=(114, 114, 114))
            # 调整bbox坐标
            if "boxes" in target and len(target["boxes"]) > 0:
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    boxes = boxes.numpy().astype(np.float32)
                    boxes[:, [0, 2]] += left
                    boxes[:, [1, 3]] += top
                    target["boxes"] = torch.from_numpy(boxes)
                else:
                    boxes = boxes.astype(np.float32)
                    boxes[:, [0, 2]] += left
                    boxes[:, [1, 3]] += top
                    target["boxes"] = boxes

        # 更新图像尺寸
        h, w = img_np.shape[:2]

        # 创建变换矩阵
        # 注意：这里简化实现，使用YOLO风格的中心变换

        # 中心
        C = np.eye(3, dtype=np.float32)
        C[0, 2] = -w / 2  # 将原点移动到中心
        C[1, 2] = -h / 2

        # 透视
        P = np.eye(3, dtype=np.float32)
        if abs(self.perspective) > 1e-6:
            P[2, 0] = random.uniform(-self.perspective, self.perspective)
            P[2, 1] = random.uniform(-self.perspective, self.perspective)

        # 旋转和缩放
        R = np.eye(3, dtype=np.float32)
        a = random.uniform(-self.degrees, self.degrees)
        s = random.uniform(1 - self.scale, 1 + self.scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

        # 剪切
        S = np.eye(3, dtype=np.float32)
        if abs(self.shear) > 1e-6:
            S[0, 1] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)
            S[1, 0] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)

        # 平移
        T = np.eye(3, dtype=np.float32)
        if abs(self.translate) > 1e-6:
            T[0, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * w
            T[1, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * h

        # 组合变换矩阵 (顺序很重要: T @ S @ R @ P @ C)
        M = T @ S @ R @ P @ C

        # 应用变换矩阵将原点移回
        M_post = np.eye(3, dtype=np.float32)
        M_post[0, 2] = w / 2
        M_post[1, 2] = h / 2
        M = M_post @ M

        # 判断是否为透视变换
        has_perspective = abs(self.perspective) > 1e-6

        # 应用变换到图像
        if has_perspective:
            img_np = cv2.warpPerspective(img_np, M, dsize=(w, h),
                                         flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT,
                                         borderValue=(114, 114, 114))
        else:
            img_np = cv2.warpAffine(img_np, M[:2], dsize=(w, h),
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=(114, 114, 114))

        # 更新边界框
        if "boxes" in target and len(target["boxes"]) > 0:
            boxes = target["boxes"]
            if isinstance(boxes, torch.Tensor):
                boxes = boxes.numpy().astype(np.float32)
            else:
                boxes = boxes.astype(np.float32)

            # 保存原始bbox用于过滤
            original_boxes = boxes.copy()

            # 应用变换到bboxes
            transformed_boxes = self.apply_bboxes(boxes, M, w, h, has_perspective)

            # 过滤无效的bbox
            if len(transformed_boxes) > 0:
                valid_indices = self.box_candidates(original_boxes, transformed_boxes)

                if valid_indices.any():
                    transformed_boxes = transformed_boxes[valid_indices]
                    target["boxes"] = torch.from_numpy(transformed_boxes)

                    # 更新所有相关字段
                    for key in ["labels", "area", "iscrowd"]:
                        if key in target:
                            val = target[key]
                            if isinstance(val, torch.Tensor):
                                val = val.numpy()
                            elif not isinstance(val, np.ndarray):
                                val = np.array(val)
                            target[key] = torch.from_numpy(val[valid_indices])
                else:
                    # 如果没有有效bbox，清空相关字段
                    target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
                    for key in ["labels", "area", "iscrowd"]:
                        if key in target:
                            if isinstance(target[key], torch.Tensor):
                                # 创建空tensor
                                if key == "labels":
                                    target[key] = torch.zeros((0,), dtype=torch.int64)
                                elif key == "area":
                                    target[key] = torch.zeros((0,), dtype=torch.float32)
                                elif key == "iscrowd":
                                    target[key] = torch.zeros((0,), dtype=torch.uint8)
            else:
                target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)

        # 转换图像格式
        if is_tensor:
            # 确保通道数正确
            if len(img_np.shape) == 2:  # 灰度图
                img_np = np.expand_dims(img_np, axis=0)
            else:
                img_np = img_np.transpose(2, 0, 1)
            image = torch.from_numpy(img_np.copy())
        else:
            image = Image.fromarray(img_np.astype(np.uint8))

        return image, target

