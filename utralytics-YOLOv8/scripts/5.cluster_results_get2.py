# 统计标注里的聚团信息

import pandas as pd
import numpy as np
import cv2
import os
from pathlib import Path

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def do_kmeans(img, k=3):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape
    datapixel = np.float32(img.reshape((h * w, 1)))  # 变为一维数据
    criteria = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 100, 1e-5)  # 设置停止条件
    flags = cv2.KMEANS_PP_CENTERS  # k-means++初始化
    _, labels, centers = cv2.kmeans(datapixel, k, None, criteria, 50, flags)
    max_index, mid_index, min_index = 0, 0, 0
    for i in range(k):  # 找到分类后各类别的标号
        if i == np.argmax(centers):
            max_index = i  # 获取最大值所在的index，对应与稀相
        elif i == np.argmin(centers):
            min_index = i  # 获取最小值所在的index，对应与密相核心区
        else:
            mid_index = i  # 对应与密相核云层
    core = datapixel[labels == min_index]
    core_cloud_threshold = np.max(core)
    result = img.copy()  # 创建输出图像的副本
    result[result > core_cloud_threshold] = 255  # 将大于阈值的像素设为255
    return core_cloud_threshold, result


def solidholdup(pixel):  # 灰度变固含率
    epsilon = 0.1 / np.log(pixel / 2.17) - 0.02  # 灰度-固含率标定曲线
    return epsilon


def corner_to_center(x1, y1, x2, y2):
    w = x2 - x1  # 计算宽度和高度
    h = y2 - y1
    cx = x1 + w / 2  # 计算中心点坐标
    cy = y1 + h / 2
    return cx, cy, w, h


def get_cluster_infor(filename, index, pred_result, img):  # 读取聚团信息
    category = pred_result[0]
    x1, y1, x2, y2 = pred_result[1]
    cx, cy, w, h = corner_to_center(x1, y1, x2, y2)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 先转换为灰度图
    gray[gray <= 3] = 3
    epsilon = solidholdup(gray)  # 转换为固含率矩阵
    epsilon[epsilon <= 0.001067] = 0  # 空白区域在固含率矩阵上更改为0
    mass = np.sum(epsilon)  # 聚团质量
    area = np.count_nonzero(epsilon)  # 聚团像素数
    area_cm2 = area * 0.015625 ** 2  # 聚团面积cm2
    avg_epsiloon = 0
    dc_cm = 0
    aspect_ratio = 0
    if area != 0:
        avg_epsiloon = mass / area  # 聚团平均固含率
        dc_cm = 2 * np.sqrt(area_cm2 / np.pi)  # 计算等效直径
        # 横纵比
        x_non_zero = np.sum(np.sum(epsilon != 0, axis=1))  # 计算每行非零元素的个数
        row_sums = np.sum(epsilon, axis=1)  # 计算每行的和
        non_zero_rows = np.sum(row_sums != 0)  # 统计非全0行的数量
        x_avg = x_non_zero / non_zero_rows if non_zero_rows != 0 else 0
        y_non_zero = np.sum(np.sum(epsilon != 0, axis=0))  # 计算每列非零元素的个数
        col_sums = np.sum(epsilon, axis=0)  # 计算每列的和
        non_zero_cols = np.sum(col_sums != 0)  # 统计非全0列的数量
        y_avg = y_non_zero / non_zero_cols if non_zero_cols != 0 else 0
        aspect_ratio = y_avg / x_avg if x_avg != 0 else 0
    characteristic = [filename, index, category, cx, cy, w, h, area_cm2, avg_epsiloon, dc_cm, aspect_ratio]
    return characteristic


def parse_yolo_label(label_path, img_width, img_height, class_names):
    """
    解析YOLO标注文件
    Args:
        label_path: 标注文件路径
        img_width: 图像宽度
        img_height: 图像高度
        class_names: 类别名称列表
    Returns:
        List of [class_name, (x1, y1, x2, y2)]
    """
    results = []
    if not os.path.exists(label_path):
        return results
    with open(label_path, 'r') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        # YOLO标准格式：class_id x_center y_center width height
        if len(parts) >= 5:
            try:
                class_id = int(parts[0])
                # 解析YOLO格式的归一化坐标
                x_center = float(parts[1]) * img_width
                y_center = float(parts[2]) * img_height
                width = float(parts[3]) * img_width
                height = float(parts[4]) * img_height
                # 计算边界框坐标
                x1 = int(x_center - width / 2)
                y1 = int(y_center - height / 2)
                x2 = int(x_center + width / 2)
                y2 = int(y_center + height / 2)
                # 确保坐标在图像范围内
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(img_width - 1, x2)
                y2 = min(img_height - 1, y2)
                # 获取类别名称
                if class_id < len(class_names):
                    category = class_names[class_id]
                else:
                    category = f"class_{class_id}"
                results.append([category, (x1, y1, x2, y2)])
            except (ValueError, IndexError) as e:
                print(f"解析标注文件 {label_path} 第{i + 1}行时出错: {e}")
    return results


def process_label_and_image(img_path, label_path, imgname, class_names):
    img = cv2.imread(img_path)
    if img is None:
        print(f"无法读取图像: {img_path}")
        return []
    img_height, img_width = 800, 1280   # TODO
    # 解析标注文件
    label_results = parse_yolo_label(label_path, img_width, img_height, class_names)
    all_cluster_results = []
    for i, label_result in enumerate(label_results):
        category, (x1, y1, x2, y2) = label_result
        # 从原始图像截取聚团区域
        cluster_img = img[y1:y2, x1:x2]
        # 保存k-means图像
        _, k_means_pic = do_kmeans(cluster_img)
        # 获取聚团信息
        cluster_info = get_cluster_infor(imgname, i, [category, (x1, y1, x2, y2)], cluster_img)
        all_cluster_results.append(cluster_info)
    return all_cluster_results


def process_directory(input_dir, labels_dir, output_excel="cluster_results_from_labels.xlsx", class_names=None):
    """
    处理整个目录
    Args:
        input_dir: 图像目录
        labels_dir: 标注文件目录
        output_excel: 输出Excel文件名
        class_names: 类别名称列表，默认为["U-type", "inverted-U", "chain", "special"]
    """
    if class_names is None:
        class_names = ["U-type", "inverted-U", "chain", "special"]
    # 支持常见的图像格式
    valid_extensions = {'.bmp', '.jpg', '.jpeg', '.png', '.tiff', '.tif'}
    # 获取目录中所有图像文件
    image_files = []
    for file_path in Path(input_dir).iterdir():
        if file_path.is_file() and file_path.suffix.lower() in valid_extensions:
            image_files.append(file_path)
    if not image_files:
        print(f"在目录 {input_dir} 中没有找到图像文件")
        return
    print(f"找到 {len(image_files)} 个图像文件，开始处理...")
    all_results = []
    for i, img_path in enumerate(image_files):
        print(f"处理进度: {i + 1}/{len(image_files)} - {img_path.name}")
        try:
            label_filename = img_path.stem + ".txt"
            label_path = Path(labels_dir) / label_filename
            if not label_path.exists():
                print(f"警告: 找不到标注文件 {label_path}，跳过图像 {img_path.name}")
                continue
            # 处理图像和标注
            cluster_results = process_label_and_image(
                str(img_path),
                str(label_path),
                img_path.name,
                class_names
            )
            all_results.extend(cluster_results)
        except Exception as e:
            print(f"处理图像 {img_path} 时出错: {str(e)}")
            continue

    if all_results:
        columns = ["filename", "index", "category", "cx", "cy", "w", "h", 'area_cm2', 'avg_epsiloon', 'dc_cm',
                   'aspect_ratio']
        results_df = pd.DataFrame(all_results, columns=columns)
        # 按文件名和索引排序
        results_df = results_df.sort_values(by=['filename', 'index'])
        with pd.ExcelWriter(output_excel, engine="xlsxwriter") as writer:
            results_df.to_excel(writer, index=False)
        print(f"处理完成！共处理 {len(all_results)} 个检测结果，已保存到 {output_excel}")
        # 打印统计信息
        print("\n=== 统计信息 ===")
        print(f"总聚团数: {len(all_results)}")
        print("各类别聚团数量:")
        category_counts = results_df['category'].value_counts()
        for category, count in category_counts.items():
            print(f"  {category}: {count}")
        print("\n平均面积 (cm²):")
        avg_area_by_category = results_df.groupby('category')['area_cm2'].mean()
        for category, avg_area in avg_area_by_category.items():
            print(f"  {category}: {avg_area:.4f}")
    else:
        print("没有生成任何检测结果")


def main():
    # 设置路径
    input_directory = r"D:\project_SJ\YOLOv3\pic-dataset-detect\cluster_dataset_subset\pic"  # 图像目录  # TODO
    labels_directory = r"D:\project_SJ\YOLOv3\pic-dataset-detect\cluster_dataset_subset\labels_v10_four"  # YOLO标注文件目录   # TODO
    output_file = "cluster_results_from_labels.xlsx"
    # 处理整个目录
    process_directory(input_directory, labels_directory, output_file)


if __name__ == '__main__':
    main()