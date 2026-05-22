# 查看所有层的特征图，受内存限制单次处理图像较少

import warnings

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import torch, yaml, cv2, os, shutil, sys, copy
import numpy as np
from pytorch_grad_cam import GradCAMPlusPlus, GradCAM, XGradCAM, EigenCAM, HiResCAM, LayerCAM, RandomCAM, EigenGradCAM, \
    KPCA_CAM, AblationCAM, ScoreCAM

np.random.seed(0)
from ultralytics.nn.tasks import attempt_load_weights
from ultralytics.utils.torch_utils import intersect_dicts
import matplotlib.pyplot as plt
from tqdm import trange
from PIL import Image
from ultralytics import YOLO
from ultralytics.utils.ops import xywh2xyxy, non_max_suppression
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients


# 用于图像预处理，主要功能是将图像调整到指定大小，同时保持原始图像的宽高比
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    shape = im.shape[:2]   # 获取原始图形尺寸
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])   # 获取缩放比例
    if not scaleup:
        r = min(r, 1.0)

    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))   # 缩放后的原图大小
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]   # 要填充的大小
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)   # 确保填充后的图能被stride整除
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]   # 直接拉伸填充整个目标区域

    dw /= 2    # 将填充平均分配到两边
    dh /= 2

    if shape[::-1] != new_unpad:   #翻转数组的高和宽
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, ratio, (top, bottom, left, right)

# 用于捕获神经网络中间层的激活值和梯度
class ActivationsAndGradients:
    def __init__(self, model, target_layers, reshape_transform):
        self.model = model
        self.gradients = []  # 存储梯度
        self.activations = []  # 存储激活值
        self.reshape_transform = reshape_transform  # 特征重塑函数
        self.handles = []   # 存储hook句柄，便于后续释放
        for target_layer in target_layers:
            self.handles.append(target_layer.register_forward_hook(self.save_activation))   # PyTorch的hook机制，前向传播时触发
            self.handles.append(target_layer.register_forward_hook(self.save_gradient))

    def save_activation(self, module, input, output):    # 捕获目标层的输出（激活图
        activation = output
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())   # 存储到CPU内存，避免GPU内存溢出

    def save_gradient(self, module, input, output):
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            return

        def _store_grad(grad):
            if self.reshape_transform is not None:
                grad = self.reshape_transform(grad)
            self.gradients = [grad.cpu().detach()] + self.gradients

        output.register_hook(_store_grad)   # 注册反向hook

    def post_process(self, result):
        if self.model.end2end:
            logits_ = result[:, :, 4:]
            boxes_ = result[:, :, :4]
            sorted, indices = torch.sort(logits_[:, :, 0], descending=True)
            return logits_[0][indices[0]], boxes_[0][indices[0]]
        elif self.model.task == 'detect':
            logits_ = result[:, 4:]     # [num_boxes, num_classes]
            boxes_ = result[:, :4]    # [num_boxes, 4]
            sorted, indices = torch.sort(logits_.max(1)[0], descending=True)    # 按类别置信度进行排序
            return torch.transpose(logits_[0], dim0=0, dim1=1)[indices[0]], torch.transpose(boxes_[0], dim0=0, dim1=1)[
                indices[0]]    # 转换维度，获取最高分框的所有类别得分，和位置坐标
        elif self.model.task == 'segment':
            logits_ = result[0][:, 4:4 + self.model.nc]
            boxes_ = result[0][:, :4]
            mask_p, mask_nm = result[1][2].squeeze(), result[1][1].squeeze().transpose(1, 0)
            c, h, w = mask_p.size()
            mask = (mask_nm @ mask_p.view(c, -1))
            sorted, indices = torch.sort(logits_.max(1)[0], descending=True)
            return torch.transpose(logits_[0], dim0=0, dim1=1)[indices[0]], torch.transpose(boxes_[0], dim0=0, dim1=1)[
                indices[0]], mask[indices[0]]
        elif self.model.task == 'pose':
            logits_ = result[:, 4:4 + self.model.nc]
            boxes_ = result[:, :4]
            poses_ = result[:, 4 + self.model.nc:]
            sorted, indices = torch.sort(logits_.max(1)[0], descending=True)
            return torch.transpose(logits_[0], dim0=0, dim1=1)[indices[0]], torch.transpose(boxes_[0], dim0=0, dim1=1)[
                indices[0]], torch.transpose(poses_[0], dim0=0, dim1=1)[indices[0]]
        elif self.model.task == 'obb':
            logits_ = result[:, 4:4 + self.model.nc]
            boxes_ = result[:, :4]
            angles_ = result[:, 4 + self.model.nc:]
            sorted, indices = torch.sort(logits_.max(1)[0], descending=True)
            return torch.transpose(logits_[0], dim0=0, dim1=1)[indices[0]], torch.transpose(boxes_[0], dim0=0, dim1=1)[
                indices[0]], torch.transpose(angles_[0], dim0=0, dim1=1)[indices[0]]
        elif self.model.task == 'classify':
            return result[0]

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        model_output = self.model(x)
        if self.model.task == 'detect':
            post_result, pre_post_boxes = self.post_process(model_output[0])
            return [[post_result, pre_post_boxes]]
        elif self.model.task == 'segment':
            post_result, pre_post_boxes, pre_post_mask = self.post_process(model_output)
            return [[post_result, pre_post_boxes, pre_post_mask]]
        elif self.model.task == 'pose':
            post_result, pre_post_boxes, pre_post_pose = self.post_process(model_output[0])
            return [[post_result, pre_post_boxes, pre_post_pose]]
        elif self.model.task == 'obb':
            post_result, pre_post_boxes, pre_post_angle = self.post_process(model_output[0])
            return [[post_result, pre_post_boxes, pre_post_angle]]
        elif self.model.task == 'classify':
            data = self.post_process(model_output)
            return [data]

    def release(self):
        for handle in self.handles:
            handle.remove()


class yolo_detect_target(torch.nn.Module):
    def __init__(self, ouput_type, conf, ratio, end2end) -> None:
        super().__init__()
        self.ouput_type = ouput_type
        self.conf = conf
        self.ratio = ratio
        self.end2end = end2end

    def forward(self, data):
        post_result, pre_post_boxes = data
        result = []
        for i in trange(int(post_result.size(0) * self.ratio)):   # 遍历检测结果（按比例）
            if (self.end2end and float(post_result[i, 0]) < self.conf) or (
                    not self.end2end and float(post_result[i].max()) < self.conf):
                break
            if self.ouput_type == 'class' or self.ouput_type == 'all':
                if self.end2end:
                    result.append(post_result[i, 0])
                else:
                    result.append(post_result[i].max())
            elif self.ouput_type == 'box' or self.ouput_type == 'all':
                for j in range(4):
                    result.append(pre_post_boxes[i, j])
        return sum(result)


class yolo_segment_target(yolo_detect_target):
    def __init__(self, ouput_type, conf, ratio, end2end):
        super().__init__(ouput_type, conf, ratio, end2end)

    def forward(self, data):
        post_result, pre_post_boxes, pre_post_mask = data
        result = []
        for i in trange(int(post_result.size(0) * self.ratio)):
            if float(post_result[i].max()) < self.conf:
                break
            if self.ouput_type == 'class' or self.ouput_type == 'all':
                result.append(post_result[i].max())
            elif self.ouput_type == 'box' or self.ouput_type == 'all':
                for j in range(4):
                    result.append(pre_post_boxes[i, j])
            elif self.ouput_type == 'segment' or self.ouput_type == 'all':
                result.append(pre_post_mask[i].mean())
        return sum(result)


class yolo_pose_target(yolo_detect_target):
    def __init__(self, ouput_type, conf, ratio, end2end):
        super().__init__(ouput_type, conf, ratio, end2end)

    def forward(self, data):
        post_result, pre_post_boxes, pre_post_pose = data
        result = []
        for i in trange(int(post_result.size(0) * self.ratio)):
            if float(post_result[i].max()) < self.conf:
                break
            if self.ouput_type == 'class' or self.ouput_type == 'all':
                result.append(post_result[i].max())
            elif self.ouput_type == 'box' or self.ouput_type == 'all':
                for j in range(4):
                    result.append(pre_post_boxes[i, j])
            elif self.ouput_type == 'pose' or self.ouput_type == 'all':
                result.append(pre_post_pose[i].mean())
        return sum(result)


class yolo_obb_target(yolo_detect_target):
    def __init__(self, ouput_type, conf, ratio, end2end):
        super().__init__(ouput_type, conf, ratio, end2end)

    def forward(self, data):
        post_result, pre_post_boxes, pre_post_angle = data
        result = []
        for i in trange(int(post_result.size(0) * self.ratio)):
            if float(post_result[i].max()) < self.conf:
                break
            if self.ouput_type == 'class' or self.ouput_type == 'all':
                result.append(post_result[i].max())
            elif self.ouput_type == 'box' or self.ouput_type == 'all':
                for j in range(4):
                    result.append(pre_post_boxes[i, j])
            elif self.ouput_type == 'obb' or self.ouput_type == 'all':
                result.append(pre_post_angle[i])
        return sum(result)


class yolo_classify_target(yolo_detect_target):
    def __init__(self, ouput_type, conf, ratio, end2end):
        super().__init__(ouput_type, conf, ratio, end2end)

    def forward(self, data):
        return data.max()


class yolo_heatmap:
    def __init__(self, weight, device, method, backward_type, conf_threshold, ratio, show_result, renormalize,
                 task, img_size):
        device = torch.device(device)
        model_yolo = YOLO(weight)
        model_names = model_yolo.names
        print(f'model class info:{model_names}')
        model = copy.deepcopy(model_yolo.model)
        model.to(device)
        model.info()
        for p in model.parameters():
            p.requires_grad_(True)
        model.eval()

        model.task = task
        if not hasattr(model, 'end2end'):
            model.end2end = False

        if task == 'detect':
            target = yolo_detect_target(backward_type, conf_threshold, ratio, model.end2end)
        elif task == 'segment':
            target = yolo_segment_target(backward_type, conf_threshold, ratio, model.end2end)
        elif task == 'pose':
            target = yolo_pose_target(backward_type, conf_threshold, ratio, model.end2end)
        elif task == 'obb':
            target = yolo_obb_target(backward_type, conf_threshold, ratio, model.end2end)
        elif task == 'classify':
            target = yolo_classify_target(backward_type, conf_threshold, ratio, model.end2end)
        else:
            raise Exception(f"not support task({task}).")

        self.model = model
        self.model_yolo = model_yolo
        self.model_names = model_names
        self.device = device
        self.target = target
        self.method_name = method
        self.backward_type = backward_type
        self.conf_threshold = conf_threshold
        self.ratio = ratio
        self.show_result = show_result
        self.renormalize = renormalize
        self.task = task
        self.img_size = img_size

        # 基于YOLOv8架构定义可用的特征层
        self.available_layers = self.get_yolov8_feature_layers()
        print(f"\n发现 {len(self.available_layers)} 个可用的特征层:")
        for layer_type, layers in self.available_layers.items():
            print(f"  {layer_type}: {layers}")

        colors = np.random.uniform(0, 255, size=(len(model_names), 3)).astype(np.int32)
        self.colors = colors

    def get_yolov8_feature_layers(self):
        """根据YOLOv8架构获取所有可用于特征可视化的层"""

        # YOLOv8架构层索引定义
        backbone_layers = {
            'Conv_Stem': [0, 1, 3, 5, 7],  # 下采样卷积层
            'Backbone_C2f': [2, 4, 6, 8],  # Backbone中的C2f层
            'SPPF': [9],  # SPPF层
        }

        head_layers = {
            'Upsample': [10, 13],  # 上采样层
            'Concat': [11, 14, 17, 20],  # 拼接层
            'Conv' : [16, 19],
            'Head_C2f': [12, 15, 18, 21],  # Head中的C2f层（重要特征层）
        }

        # 特别标注三个尺度的输出层（用于Detect）
        scale_outputs = {
            'P3_Small_Scale': [15],  # P3/8-small
            'P4_Medium_Scale': [18],  # P4/16-medium
            'P5_Large_Scale': [21],  # P5/32-large
        }

        # 合并所有层，但排除Detect层(22)
        all_layers = {}
        all_layers.update(backbone_layers)
        all_layers.update(head_layers)
        all_layers.update(scale_outputs)

        # 打印层描述信息
        layer_descriptions = {
            0: 'Conv (P1/2)',
            1: 'Conv (P2/4)',
            2: 'C2f (Backbone)',
            3: 'Conv (P3/8)',
            4: 'C2f (Backbone)',
            5: 'Conv (P4/16)',
            6: 'C2f (Backbone)',
            7: 'Conv (P5/32)',
            8: 'C2f (Backbone)',
            9: 'SPPF',
            10: 'Upsample',
            11: 'Concat',
            12: 'C2f (Head P4)',
            13: 'Upsample',
            14: 'Concat',
            15: 'C2f (Head P3/8-small) ★',
            16: 'Conv',
            17: 'Concat',
            18: 'C2f (Head P4/16-medium) ★',
            19: 'Conv',
            20: 'Concat',
            21: 'C2f (Head P5/32-large) ★',
            22: 'Detect (skip)'
        }

        print("\nYOLOv8层结构:")
        for i in range(23):
            desc = layer_descriptions.get(i, f'Layer {i}')
            if i in [15, 18, 21]:
                print(f"  {i:2d}: {desc} ← 推荐用于GradCAM")
            elif i == 22:
                print(f"  {i:2d}: {desc}")
            else:
                print(f"  {i:2d}: {desc}")

        return all_layers

    def get_prediction_results(self, tensor):
        """安全地获取预测结果"""
        try:
            pred = self.model_yolo.predict(tensor, conf=self.conf_threshold, iou=0.7, verbose=False)[0]
            return pred, None
        except Exception as e:
            return None, str(e)


    def process_single_layer(self, img, tensor, layer_idx, save_path, img_name, layer_desc=""):
        """处理单个层的特征图"""
        try:
            # 设置目标层
            target_layers = [self.model.model[layer_idx]]

            # 重新初始化CAM方法
            method = eval(self.method_name)(self.model, target_layers)
            method.activations_and_grads = ActivationsAndGradients(self.model, target_layers, None)

            # 生成CAM
            grayscale_cam = method(tensor, [self.target])
            grayscale_cam = grayscale_cam[0, :]

            # 生成热力图
            cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)

            # 安全地获取预测结果
            pred, error = self.get_prediction_results(tensor)

            # 如果需要重归一化
            if self.renormalize and self.task in ['detect', 'segment', 'pose'] and pred is not None:
                try:
                    if hasattr(pred, 'boxes') and pred.boxes is not None:
                        boxes = pred.boxes.xyxy.cpu().detach().numpy().astype(np.int32)
                        cam_image = self.renormalize_cam_in_bounding_boxes(boxes, img, grayscale_cam)
                except Exception:
                    pass

            # 如果需要显示检测结果
            if self.show_result and pred is not None:
                try:
                    cam_image = pred.plot(img=cam_image,
                                          conf=None,
                                          font_size=None,
                                          line_width=None,
                                          labels=None)
                except Exception:
                    pass

            heatmap = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

            # 保存纯热力图
            if layer_desc:
                heatmap_filename = f"{img_name}_layer{layer_idx}_{layer_desc}_heatmap.png"
            else:
                heatmap_filename = f"{img_name}_layer{layer_idx}_heatmap.png"
            heatmap_filepath = os.path.join(save_path, heatmap_filename)

            # 转换为0-255范围并保存
            heatmap_pil = Image.fromarray(heatmap)
            heatmap_pil.save(heatmap_filepath)
            print(f"  已保存热力图: {heatmap_filename}")

            # 保存图像 - 添加层描述
            if layer_desc:
                save_filename = f"{img_name}_layer{layer_idx}_{layer_desc}.png"
            else:
                save_filename = f"{img_name}_layer{layer_idx}.png"

            save_filepath = os.path.join(save_path, save_filename)

            # 转换为PIL并保存
            cam_image_pil = Image.fromarray(cam_image)
            cam_image_pil.save(save_filepath)
            print(f"  已保存: {save_filename}")
            # # ===== 新增：保存带colorbar的图像 =====
            # try:
            #     import matplotlib.pyplot as plt
            #     from matplotlib.colors import Normalize
            #     cbar_filepath = save_filepath.replace('.png', '_with_cbar.png')
            #     plt.figure(figsize=(10, 8))
            #     plt.imshow(cam_image)
            #     plt.axis('off')
            #     plt.title(f'Grad-CAM Layer {layer_idx} {layer_desc}')
            #     # 创建colorbar
            #     norm = Normalize(vmin=0, vmax=1)
            #     sm = plt.cm.ScalarMappable(cmap='jet', norm=norm)
            #     sm.set_array([])
            #     cbar = plt.colorbar(sm, fraction=0.03, pad=0.03)
            #     cbar.set_label('Activation Intensity', rotation=270, labelpad=15)
            #     plt.savefig(cbar_filepath, dpi=300, bbox_inches='tight', pad_inches=0.1)
            #     plt.close()
            #
            #     print(f"  已保存带colorbar: {os.path.basename(cbar_filepath)}")
            # except Exception as e:
            #     print(f"  保存colorbar失败: {e}")



        except Exception as e:
            print(f"  处理层 {layer_idx} 失败: {str(e)[:100]}...")

    def process_image(self, img_path, base_save_dir):
        """处理单张图像，生成所有层的特征图"""
        try:
            img = cv2.imdecode(np.fromfile(img_path, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                print(f"警告: {img_path} 读取失败。")
                return
        except Exception as e:
            print(f"警告: {img_path} 读取失败: {e}")
            return

        img_name = os.path.splitext(os.path.basename(img_path))[0]

        image_save_dir = os.path.join(base_save_dir, img_name)
        os.makedirs(image_save_dir, exist_ok=True)
        print(f"\n处理图像: {img_name}")
        print(f"保存到: {image_save_dir}")

        # 图像预处理
        img_resized, _, (top, bottom, left, right) = letterbox(
            img, new_shape=(self.img_size, self.img_size), auto=True)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_normalized = np.float32(img_rgb) / 255.0
        tensor = torch.from_numpy(np.transpose(img_normalized, axes=[2, 0, 1])).unsqueeze(0).to(self.device)

        # 遍历所有层
        successful_layers = 0
        layer_descriptions = {
            15: 'P3_small',
            18: 'P4_medium',
            21: 'P5_large',
        }

        # 从available_layers中提取所有层索引
        all_layer_indices = []
        for layer_type, layers in self.available_layers.items():
            all_layer_indices.extend(layers)
        all_layer_indices = sorted(list(set(all_layer_indices)))  # 去重并排序

        print(f"\n将处理 {len(all_layer_indices)} 个特征层: {all_layer_indices}")

        for layer_idx in all_layer_indices:
            layer_desc = layer_descriptions.get(layer_idx, "")
            self.process_single_layer(img_normalized, tensor, layer_idx, image_save_dir, img_name, layer_desc)
            successful_layers += 1

        if successful_layers == 0:
            print(f"  警告: 图像 {img_name} 所有层都处理失败")

    def renormalize_cam_in_bounding_boxes(self, boxes, image_float_np, grayscale_cam):
        renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
        for x1, y1, x2, y2 in boxes:
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(grayscale_cam.shape[1] - 1, x2), min(grayscale_cam.shape[0] - 1, y2)
            renormalized_cam[y1:y2, x1:x2] = scale_cam_image(grayscale_cam[y1:y2, x1:x2].copy())
        renormalized_cam = scale_cam_image(renormalized_cam)
        eigencam_image_renormalized = show_cam_on_image(image_float_np, renormalized_cam, use_rgb=True)
        return eigencam_image_renormalized

    def __call__(self, img_dir, save_base_dir):
        os.makedirs(save_base_dir, exist_ok=True)

        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []

        for file in os.listdir(img_dir):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(file)

        if not image_files:
            print(f"错误: 在 {img_dir} 中没有找到图像文件")
            return

        # 获取所有层索引
        all_layer_indices = []
        for layer_type, layers in self.available_layers.items():
            all_layer_indices.extend(layers)
        all_layer_indices = sorted(list(set(all_layer_indices)))

        print(f"\n找到 {len(image_files)} 张图像")
        print(f"将处理 {len(all_layer_indices)} 个特征层: {all_layer_indices}")
        print(f"总共将生成 {len(image_files) * len(all_layer_indices)} 张特征图")

        for i, img_file in enumerate(image_files):
            print(f"\n--- 处理第 {i + 1}/{len(image_files)} 张图像 ---")
            img_path = os.path.join(img_dir, img_file)
            self.process_image(img_path, save_base_dir)

        print(f"\n处理完成！所有特征图已保存到: {save_base_dir}")


def get_params():
    params = {
        'weight': r'D:\project_SJ\CUP-cluster-detection\utralytics\scripts\runs\3.final_augment_model\weights\best.pt',
        'device': 'cuda:0',
        'method': 'GradCAMPlusPlus',
        'backward_type': 'all',
        'conf_threshold': 0.2,
        'ratio': 0.02,
        'show_result': False,
        'renormalize': False,
        'task': 'detect',
        'img_size': 640,
    }
    return params


if __name__ == '__main__':
    model = yolo_heatmap(**get_params())

    input_dir = r'C:\Users\Administrator\Desktop\1111'
    output_dir = r'C:\Users\Administrator\Desktop\1111\GradCAMPlusPlus'

    model(input_dir, output_dir)