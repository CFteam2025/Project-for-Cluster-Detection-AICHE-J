from ultralytics import YOLO
import torch
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def draw_box_and_label(image, xyxy, cls_name, conf, color, top_extension, right_extension):
    x1, y1, x2, y2 = xyxy
    # 调整坐标：由于图像扩展了顶部和右侧，但原始坐标已经偏移，所以这里只需要处理文字显示位置
    cls_name = 'U-type' if cls_name == 'U-shape' else "chain" if cls_name == "long-chain" else cls_name
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
    # 计算文本大小
    text = f"{cls_name}{conf:.2f}"
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2, 4)
    # 确保文本不会超出图像顶部（考虑顶部扩展）
    text_y = max(y1 - 15, text_height + 10)
    # 确保文本不会超出图像右侧（考虑右侧扩展）
    # 如果文本太靠近右侧边界，向左移动
    max_x = image.shape[1] - text_width - 10  # 右侧留出10像素边距
    text_x = min(x1, max_x)  # 取x1和最大允许位置中的较小值
    # 如果检测框在图像边缘，确保文字完全可见
    if x1 + text_width > image.shape[1]:
        text_x = image.shape[1] - text_width - 10
    cv2.putText(image, text, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4)
    return image


def extend_image_sides(image, top_extension=100, right_extension=200, background_color=(255, 255, 255)):
    """
    在图像顶部和右侧扩展指定高度的区域
    """
    height, width = image.shape[:2]
    # 创建顶部扩展区域
    top_extension_region = np.full((top_extension, width, 3), background_color, dtype=np.uint8)
    # 创建右侧扩展区域（包括顶部扩展部分）
    right_extension_region = np.full((height + top_extension, right_extension, 3), background_color, dtype=np.uint8)
    # 先垂直拼接顶部扩展
    extended_image = np.vstack((top_extension_region, image))
    # 再水平拼接右侧扩展
    extended_image = np.hstack((extended_image, right_extension_region))
    return extended_image, top_extension, right_extension


def adjust_boxes_for_extension(boxes, top_extension, right_extension):
    """
    调整边界框坐标以适应扩展后的图像
    """
    adjusted_boxes = []
    for box in boxes:
        x1, y1, x2, y2 = box
        # y坐标向下移动（顶部扩展）
        # x坐标保持不变（右侧扩展不需要移动x坐标，因为是在右侧添加空间）
        adjusted_boxes.append([x1, y1 + top_extension, x2, y2 + top_extension])
    return adjusted_boxes


def process_image(model, image_path, output_dir, visualize=False):
    try:
        large_img = cv2.imread(image_path)
        if large_img is None:
            raise FileNotFoundError(f"Failed to read the image: {image_path}. Please check the file path.")
    except FileNotFoundError as e:
        print(e)
        return
    # 在图像顶部和右侧扩展区域
    top_extension = 50  # 顶部扩展高度
    right_extension = 100  # 右侧扩展宽度
    extended_img, top_extension, right_extension = extend_image_sides(large_img, top_extension, right_extension)
    colormap = {
        "U-type": (255, 0, 0),
        "inverted-U": (0, 255, 0),
        "chain": (0, 0, 255),
        "special": (255, 255, 0),
    }
    results = model.predict(large_img, conf=0.2, iou=0.2, visualize=visualize, device='cuda:0', agnostic_nms=True)
    for result in results:
        boxes = result.boxes
        names = result.names
        num = len(boxes.cls.cpu().numpy().astype(int))  # 框数
        if num >= 1:
            for i in range(num):
                xyxy = boxes.xyxy.cpu().numpy().astype(int)[i]
                cls = boxes.cls.cpu().numpy().astype(int)[i]
                conf = boxes.conf.cpu().numpy()[i]
                color = colormap.get(names.get(cls), (0, 255, 0))
                color = (color[2], color[1], color[0])  # 将RGB格式转换为BGR
                # 调整边界框坐标以适应扩展后的图像（只调整y坐标）
                x1, y1, x2, y2 = xyxy
                y1 += top_extension
                y2 += top_extension
                # x坐标保持不变，因为右侧扩展是在原始图像右侧添加空间
                extended_img = draw_box_and_label(
                    extended_img,
                    (x1, y1, x2, y2),
                    names.get(cls),
                    conf,
                    color,
                    top_extension,
                    right_extension
                )
    image_name = os.path.basename(image_path)
    output_path = os.path.join(output_dir, image_name)
    cv2.imwrite(output_path, extended_img)
    print(output_path)
    plt.close()


def my_predict():
    model = YOLO(
        r"D:\project_SJ\CUP-cluster-detection\utralytics\scripts\runs\3.final_augment_model\weights\best.pt")
    input_dir = r"C:\Users\Administrator\Desktop\2222"
    output_dir = r"C:\Users\Administrator\Desktop\best-pred"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_extensions = ['.bmp', '.jpg', '.jpeg', '.png']

    # 只处理指定目录，不递归
    for file in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file)

        # 跳过子目录
        if not os.path.isfile(file_path):
            continue

        # 跳过输出目录
        if file_path.startswith(output_dir):
            continue

        if any(file.lower().endswith(ext) for ext in image_extensions):
            process_image(model, file_path, output_dir)


if __name__ == '__main__':
    my_predict()