import pandas as pd
import numpy as np
import cv2
import os
from pathlib import Path

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def do_kmeans(img, k=3):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape
    datapixel = np.float32(img.reshape((h * w, 1)))
    criteria = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
    flags = cv2.KMEANS_PP_CENTERS
    _, labels, centers = cv2.kmeans(datapixel, k, None, criteria, 50, flags)

    min_index = np.argmin(centers)
    core = datapixel[labels == min_index]
    core_cloud_threshold = np.max(core)

    result = img.copy()
    result[result > core_cloud_threshold] = 255
    return core_cloud_threshold, result


def solidholdup(pixel):
    epsilon = 0.1 / np.log(pixel / 2.17) - 0.02
    return epsilon


def corner_to_center(x1, y1, x2, y2):
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2
    cy = y1 + h / 2
    return cx, cy, w, h


def get_cluster_infor(filename, label_id, cx, img_width, img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray[gray <= 3] = 3

    epsilon = solidholdup(gray)
    epsilon[epsilon <= 0.001067] = 0

    mass = np.sum(epsilon)
    area = np.count_nonzero(epsilon)
    area_cm2 = area * 0.015625 ** 2

    avg_epsiloon = 0
    dc_cm = 0
    aspect_ratio = 0

    if area != 0:
        avg_epsiloon = mass / area
        dc_cm = 2 * np.sqrt(area_cm2 / np.pi)

        x_non_zero = np.sum(np.sum(epsilon != 0, axis=1))
        non_zero_rows = np.sum(np.sum(epsilon, axis=1) != 0)
        x_avg = x_non_zero / non_zero_rows if non_zero_rows != 0 else 0

        y_non_zero = np.sum(np.sum(epsilon != 0, axis=0))
        non_zero_cols = np.sum(np.sum(epsilon, axis=0) != 0)
        y_avg = y_non_zero / non_zero_cols if non_zero_cols != 0 else 0

        aspect_ratio = y_avg / x_avg if x_avg != 0 else 0

    xc_norm = cx / img_width

    return [
        filename,
        label_id,
        xc_norm,
        area_cm2,
        avg_epsiloon,
        dc_cm,
        aspect_ratio
    ]


def parse_yolo_label(label_path, img_width, img_height, class_names):
    results = []
    if not os.path.exists(label_path):
        return results

    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        class_id = int(parts[0])
        x_center = float(parts[1]) * img_width
        y_center = float(parts[2]) * img_height
        width = float(parts[3]) * img_width
        height = float(parts[4]) * img_height

        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_width - 1, x2)
        y2 = min(img_height - 1, y2)

        category = class_names[class_id]
        results.append((class_id, x1, y1, x2, y2))

    return results


def process_label_and_image(img_path, label_path, imgname, class_names):
    img = cv2.imread(img_path)
    if img is None:
        return []

    img_height, img_width = img.shape[:2]

    labels = parse_yolo_label(label_path, img_width, img_height, class_names)
    results = []

    for label_id, x1, y1, x2, y2 in labels:
        cluster_img = img[y1:y2, x1:x2]
        if cluster_img.size == 0:
            continue

        cx, cy, w, h = corner_to_center(x1, y1, x2, y2)

        cluster_info = get_cluster_infor(
            imgname,
            label_id,
            cx,
            img_width,
            cluster_img
        )

        results.append(cluster_info)

    return results


def process_directory(input_dir, labels_dir, output_excel, class_names):
    valid_ext = {'.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff'}
    image_files = [p for p in Path(input_dir).iterdir()
                   if p.suffix.lower() in valid_ext]

    all_results = []

    for img_path in image_files:
        label_path = Path(labels_dir) / (img_path.stem + ".txt")
        if not label_path.exists():
            continue

        res = process_label_and_image(
            str(img_path),
            str(label_path),
            img_path.name,
            class_names
        )
        all_results.extend(res)

    if not all_results:
        print("没有结果")
        return

    columns = [
        "filename",
        "label",
        "xc",
        "area_cm2",
        "avg_epsiloon",
        "dc_cm",
        "aspect_ratio"
    ]

    df = pd.DataFrame(all_results, columns=columns)

    area_bins = [
        (0, 1, "0-1"),
        (1, 2, "1-2"),
        (2, 3, "2-3"),
        (3, 4, "3-4"),
        (4, 5, "4-5"),
        (5, 6, "5-6"),
        (6, 7, "6-7"),
        (7, 8, "7-8"),
        (8, 9, "8-9"),
        (9, 10, "9-10"),
        (10, np.inf, ">10"),
    ]

    with pd.ExcelWriter(output_excel, engine="xlsxwriter") as writer:
        for low, high, sheet in area_bins:
            if high == np.inf:
                df_bin = df[df["area_cm2"] >= low]
            else:
                df_bin = df[(df["area_cm2"] >= low) & (df["area_cm2"] < high)]

            if not df_bin.empty:
                df_bin.to_excel(writer, sheet_name=sheet, index=False)

    print(f"处理完成，结果已保存到 {output_excel}")


def main():
    input_directory = r"D:\project_SJ\YOLOv3\pic-dataset-detect\cluster_dataset_subset\pic"
    labels_directory = r"D:\project_SJ\YOLOv3\pic-dataset-detect\cluster_dataset_subset\labels_v10_four"
    output_excel = "cluster_statistics_by_area.xlsx"

    class_names = ["U-type", "inverted-U", "chain", "special"]

    process_directory(
        input_directory,
        labels_directory,
        output_excel,
        class_names
    )


if __name__ == "__main__":
    main()

