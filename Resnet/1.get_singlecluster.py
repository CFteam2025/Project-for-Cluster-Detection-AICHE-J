# 将检测数据集中的标注结果转换为用于分类的图像数据

import os
import glob
import cv2


def get_yolo_labels(file_path, img_width=1280, img_height=800):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    label_results = []
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        try:
            float_parts = list(map(float, parts))
            if len(float_parts) < 5:
                print(f"警告：标注行格式不正确 - {line}")
                continue
            center_x, center_y, width, height = float_parts[1:5]
            x = int((center_x - width / 2) * img_width)
            y = int((center_y - height / 2) * img_height)
            w = int(width * img_width)
            h = int(height * img_height)
            x = max(0, x)
            y = max(0, y)
            w = max(1, w)
            h = max(1, h)
            new_parts = [float_parts[0], x, y, w, h]
            label_results.append(new_parts)
        except Exception as e:
            print(f"处理标注行时出错: {line}, 错误: {e}")
    return label_results


def get_cluster(class_names, filename, gray_img, label_results, output_dir):
    if gray_img is None or gray_img.size == 0:
        print("错误：输入图像为空，无法提取聚团区域")
        return
    for i in range(len(label_results)):
        try:
            class_idx = int(label_results[i][0])
            if class_idx < 0 or class_idx >= len(class_names):
                print(f"警告：无效的类别索引 {class_idx}，跳过该标注")
                continue
            # 使用指定的输出目录，而不是标签目录
            class_dir = os.path.join(output_dir, class_names[class_idx])
            x = label_results[i][1]
            y = label_results[i][2]
            w = label_results[i][3]
            h = label_results[i][4]
            img_height, img_width = gray_img.shape[:2]
            x2 = min(x + w, img_width)
            y2 = min(y + h, img_height)
            x = max(0, x)
            y = max(0, y)
            cluster_img = gray_img[y:y2, x:x2]
            if cluster_img.size == 0:
                print(f"警告：提取的聚团区域为空，跳过保存 - 索引 {i}")
                continue
            # 确保类别目录存在
            os.makedirs(class_dir, exist_ok=True)
            file_path = os.path.join(class_dir, f"{filename}-{i}.bmp")
            success = cv2.imwrite(file_path, cluster_img)
            if success:
                print(f"成功保存图像: {file_path}")
            else:
                print(f"错误：无法保存图像到 {file_path}")
        except Exception as e:
            print(f"处理聚团 {i} 时出错: {e}")
    return


def cal_operating_condition(image_dir, label_dir, output_dir):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    print(f"所有输出将保存到: {output_dir}")
    # 获取所有图像文件
    image_extensions = ['*.bmp', '*.jpg', '*.jpeg', '*.png']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_dir, ext)))

    # 读取类别名称
    classes_file = os.path.join(label_dir, "classes.txt")
    if not os.path.exists(classes_file):
        print(f"错误：类别文件不存在 - {classes_file}")
        return
    with open(classes_file, 'r', encoding='utf-8') as file:
        class_names = [line.strip() for line in file if line.strip()]
    for img_ind, img_file in enumerate(image_files, 1):     # 遍历处理每张图像
        try:
            _, filename = os.path.split(img_file)
            print(f"正在处理第{img_ind}张图片：{filename}")
            name = os.path.splitext(filename)[0]
            txt_name = name + ".txt"
            label_path = os.path.join(label_dir, txt_name)
            if not os.path.exists(label_path):
                print(f"警告：标注文件不存在 - {label_path}")
                continue
            gray_img = cv2.imread(img_file, flags=0)
            if gray_img is None:
                print(f"错误：无法读取图像 - {img_file}")
                continue
            gray_img[gray_img <= 3] = 3
            label_results = get_yolo_labels(label_path)
            if label_results:
                get_cluster(class_names, name, gray_img, label_results, output_dir)
            else:
                print(f"警告：未找到有效的标注信息 - {label_path}")
        except Exception as e:
            print(f"处理图片 {img_file} 时出错: {e}")
    return


if __name__ == "__main__":
    # 可以在这里指定输入图像目录、标签目录和输出目录
    cal_operating_condition(
        image_dir=r'D:\project_SJ\YOLOv3\pic-dataset-detect\cluster_dataset_subset\pic',
        label_dir=r"D:\project_SJ\YOLOv3\pic-dataset-detect\cluster_dataset_subset\labels_v10_four",
        output_dir=r'D:\project_SJ\Resnet50d_classification\datasets_classify\cluster_four_origin.v10'  # 指定输出目录
    )





