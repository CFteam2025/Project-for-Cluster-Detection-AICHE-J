# 统计预测的聚团信息，保存预测后的图像

import pandas as pd
from ultralytics import YOLO
import numpy as np
import cv2
import os
from pathlib import Path
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def draw_box_and_label(image, xyxy, cls_name, conf, color):
    x1, y1, x2, y2 = xyxy
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 4)
    text = f"{cls_name}{conf:.2f}"    # 计算文本大小
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 4)
    text_y = max(y1 - 15, text_height + 10)   # 确保文本不会超出图像顶部
    cv2.putText(image, text, (x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 4)
    return image


def extend_image_top(image, extension_height=100, background_color=(120, 120, 120)):
    """
    在图像顶部扩展指定高度的区域
    """
    height, width = image.shape[:2]
    extension = np.full((extension_height, width, 3), background_color, dtype=np.uint8)   # 创建扩展区域
    extended_image = np.vstack((extension, image))   # 垂直拼接扩展区域和原图
    return extended_image, extension_height


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
    w = x2 - x1   # 计算宽度和高度
    h = y2 - y1
    cx = x1 + w / 2   # 计算中心点坐标
    cy = y1 + h / 2
    return cx, cy, w, h


def get_cluster_infor(filename, index, pred_result, img):  # 读取聚团信息
    category = pred_result[0]
    x1, y1, x2, y2 = pred_result[1]
    cx, cy, w, h = corner_to_center(x1, y1, x2, y2)
    conf = pred_result[2]
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
    characteristic = [filename, index, category, cx, cy, w, h, conf, area_cm2, avg_epsiloon, dc_cm, aspect_ratio]
    return characteristic


def my_predict(model_path, img, imgname):
    output_dir = r"C:\Users\Administrator\Desktop\heatmap-pictest"
    model = YOLO(model_path)
    # 自定义框颜色
    colormap = {
        "U-type": (255, 0, 0),
        "inverted-U": (0, 255, 0),
        "chain": (0, 0, 255),
        "special": (255, 255, 0),
    }
    predict_img = img.copy()    # 创建用于预测的副本（不扩展）
    extension_height = 50  # 可以根据需要调整这个值
    draw_img, extension_height = extend_image_top(img.copy(), extension_height)
    results = model.predict(predict_img, conf=0.2, iou=0.2, agnostic_nms=True)
    pred_results = []
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
                x1 = xyxy[0]
                y1 = xyxy[1]
                x2 = xyxy[2]
                y2 = xyxy[3]
                # 首先从原始图像截取灰度图像（不包含彩色印记）
                cluster_img = img[y1:y2, x1:x2]  # 使用原始img，不是predict_img或draw_img
                _, k_means_pic = do_kmeans(cluster_img)
                # 在扩展后的绘制图像上绘制边界框（调整y坐标）
                adjusted_y1 = y1 + extension_height
                adjusted_y2 = y2 + extension_height
                draw_box_and_label(draw_img, (x1, adjusted_y1, x2, adjusted_y2), names.get(cls), conf, color)
                # 获取聚类信息（使用原始坐标）
                pred_result = get_cluster_infor(imgname, i, [names.get(cls), (x1, y1, x2, y2), conf], cluster_img)
                pred_results.append(pred_result)
    # 保存带标注的图像（扩展后的）
    annotated_output_path = os.path.join(output_dir, "annotated_" + imgname)
    cv2.imwrite(annotated_output_path, draw_img)
    print(annotated_output_path)
    return pred_results


def process_directory(input_dir, model_path, output_excel="cluster_results.xlsx"):
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
            # 读取图像
            img = cv2.imread(img_path)
            if img is None:
                print(f"警告: 无法读取图像 {img_path}，跳过")
                continue
            image_name = img_path.name
            # 进行预测
            pred_results = my_predict(model_path, img, image_name)
            # 将结果添加到总列表中
            all_results.extend(pred_results)
        except Exception as e:
            print(f"处理图像 {img_path} 时出错: {str(e)}")
            continue
    # 保存所有结果到Excel
    if all_results:
        columns = ["filename", "index", "category", "cx", "cy", "w", "h", 'conf', 'area_cm2', 'avg_epsiloon', 'dc_cm',
                   'aspect_ratio']
        results_df = pd.DataFrame(all_results, columns=columns)
        with pd.ExcelWriter(output_excel, engine="xlsxwriter") as writer:
            results_df.to_excel(writer, index=False)
        print(f"处理完成！共处理 {len(all_results)} 个检测结果，已保存到 {output_excel}")
    else:
        print("没有生成任何检测结果")


if __name__ == '__main__':
    # 设置路径
    input_directory = r"C:\Users\Administrator\Desktop\heatmap-pictest"  # 图像目录    # TODO
    model_path = r"D:\project_SJ\CUP-cluster-detection\utralytics\scripts\runs\3.final_augment_model\weights\best.pt"  # TODO
    output_file = "cluster_results.xlsx"  # TODO
    # 处理整个目录
    process_directory(input_directory, model_path, output_file)