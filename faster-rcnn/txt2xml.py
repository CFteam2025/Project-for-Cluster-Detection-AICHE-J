import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
import json


class YOLOToVOCConverter:
    def __init__(self, class_mapping, image_folder, txt_folder, output_folder,
                 dataset_name="ClusterDataSet", img_width=1280, img_height=800):
        """
        初始化转换器

        Args:
            class_mapping (dict): 类别映射 {类别名: YOLO_ID}
            image_folder (str): 图像文件夹路径
            txt_folder (str): YOLO标注文件夹路径
            output_folder (str): 输出的XML文件夹路径
            dataset_name (str): 数据集名称
            img_width (int): 图像宽度（固定）
            img_height (int): 图像高度（固定）
        """
        # 反转映射：从YOLO_ID到类别名
        self.id_to_class = {v: k for k, v in class_mapping.items()}
        self.image_folder = image_folder
        self.txt_folder = txt_folder
        self.output_folder = output_folder
        self.dataset_name = dataset_name
        self.img_width = img_width
        self.img_height = img_height

        # 创建输出文件夹
        os.makedirs(output_folder, exist_ok=True)

    def parse_yolo_line(self, line):
        """
        解析YOLO格式的一行

        Args:
            line (str): YOLO格式行 "class_id cx cy w h"

        Returns:
            dict: 包含类别和边界框信息的字典
        """
        parts = line.strip().split()
        if len(parts) != 5:
            return None

        class_id = int(parts[0])

        # 检查类别ID是否有效
        if class_id not in self.id_to_class:
            print(f"警告: 未知的类别ID {class_id}")
            return None

        # 解析归一化坐标
        cx = float(parts[1]) * self.img_width
        cy = float(parts[2]) * self.img_height
        w = float(parts[3]) * self.img_width
        h = float(parts[4]) * self.img_height

        # 计算实际坐标
        xmin = int(cx - w / 2)
        ymin = int(cy - h / 2)
        xmax = int(cx + w / 2)
        ymax = int(cy + h / 2)

        # 确保坐标在图像范围内
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(self.img_width - 1, xmax)
        ymax = min(self.img_height - 1, ymax)

        # 确保边界框有效
        if xmax <= xmin or ymax <= ymin:
            print(f"警告: 无效的边界框 ({xmin}, {ymin}, {xmax}, {ymax})")
            return None

        return {
            'class_id': class_id,
            'class_name': self.id_to_class[class_id],
            'xmin': xmin,
            'ymin': ymin,
            'xmax': xmax,
            'ymax': ymax
        }

    def create_xml_structure(self, filename):
        """
        创建XML基础结构

        Args:
            filename (str): 图像文件名

        Returns:
            xml.etree.ElementTree.Element: XML根元素
        """
        # 创建根元素
        annotation = ET.Element("annotation")

        # 文件夹
        folder = ET.SubElement(annotation, "folder")
        folder.text = "ClusterImages"

        # 文件名
        filename_elem = ET.SubElement(annotation, "filename")
        filename_elem.text = filename

        # 来源
        source = ET.SubElement(annotation, "source")
        database = ET.SubElement(source, "database")
        database.text = self.dataset_name
        annotation_elem = ET.SubElement(source, "annotation")
        annotation_elem.text = "PASCAL " + self.dataset_name
        image = ET.SubElement(source, "image")
        image.text = "custom"

        # 图像尺寸（固定）
        size = ET.SubElement(annotation, "size")
        width_elem = ET.SubElement(size, "width")
        width_elem.text = str(self.img_width)
        height_elem = ET.SubElement(size, "height")
        height_elem.text = str(self.img_height)
        depth_elem = ET.SubElement(size, "depth")
        depth_elem.text = "3"  # BMP通常是RGB三通道

        # 分割标记
        segmented = ET.SubElement(annotation, "segmented")
        segmented.text = "0"

        return annotation

    def add_object_to_xml(self, annotation, obj_info):
        """
        添加目标对象到XML

        Args:
            annotation (xml.etree.ElementTree.Element): XML根元素
            obj_info (dict): 目标信息字典
        """
        obj = ET.SubElement(annotation, "object")

        # 类别名称
        name = ET.SubElement(obj, "name")
        name.text = obj_info['class_name']

        # 姿态
        pose = ET.SubElement(obj, "pose")
        pose.text = "Unspecified"

        # 是否截断
        truncated = ET.SubElement(obj, "truncated")
        truncated.text = "0"

        # 是否困难
        difficult = ET.SubElement(obj, "difficult")
        difficult.text = "0"

        # 边界框
        bndbox = ET.SubElement(obj, "bndbox")
        xmin = ET.SubElement(bndbox, "xmin")
        xmin.text = str(obj_info['xmin'])
        ymin = ET.SubElement(bndbox, "ymin")
        ymin.text = str(obj_info['ymin'])
        xmax = ET.SubElement(bndbox, "xmax")
        xmax.text = str(obj_info['xmax'])
        ymax = ET.SubElement(bndbox, "ymax")
        ymax.text = str(obj_info['ymax'])

    def prettify_xml(self, elem):
        """
        美化XML格式

        Args:
            elem (xml.etree.ElementTree.Element): XML元素

        Returns:
            str: 格式化的XML字符串
        """
        rough_string = ET.tostring(elem, 'utf-8')
        reparsed = minidom.parseString(rough_string)

        # 移除XML声明前的空行
        xml_str = reparsed.toprettyxml(indent="\t")
        # 保持原有格式，但移除第一行的空行
        lines = xml_str.split('\n')
        if lines and lines[0].strip() == '':
            lines = lines[1:]
        return '\n'.join(lines)

    def convert_file(self, txt_filename):
        """
        转换单个文件

        Args:
            txt_filename (str): txt文件名
        """
        # 获取对应的图像文件名
        base_name = os.path.splitext(txt_filename)[0]
        image_filename = base_name + ".bmp"

        # 检查图像文件是否存在（可选）
        image_path = os.path.join(self.image_folder, image_filename)
        if not os.path.exists(image_path):
            print(f"警告: 图像文件不存在 {image_filename}，但仍将继续转换")

        # 读取YOLO标注文件
        txt_path = os.path.join(self.txt_folder, txt_filename)
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"无法读取文件 {txt_path}: {e}")
            return False

        # 创建XML结构
        annotation = self.create_xml_structure(image_filename)

        # 解析并添加每个目标
        objects_added = 0
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:  # 跳过空行
                continue

            obj_info = self.parse_yolo_line(line)
            if obj_info:
                self.add_object_to_xml(annotation, obj_info)
                objects_added += 1
            else:
                print(f"警告: {txt_filename} 第{line_num}行格式错误: {line}")

        if objects_added == 0:
            print(f"警告: {txt_filename} 中没有有效的目标")

        # 保存XML文件
        xml_filename = base_name + '.xml'
        xml_path = os.path.join(self.output_folder, xml_filename)

        try:
            pretty_xml = self.prettify_xml(annotation)
            with open(xml_path, 'w', encoding='utf-8') as f:
                f.write(pretty_xml)
            return True
        except Exception as e:
            print(f"无法保存XML文件 {xml_path}: {e}")
            return False

    def convert_all(self):
        """
        批量转换所有txt文件
        """
        if not os.path.exists(self.txt_folder):
            print(f"错误: 标注文件夹不存在 {self.txt_folder}")
            return

        # 获取所有txt文件
        txt_files = [f for f in os.listdir(self.txt_folder)
                     if f.lower().endswith('.txt') and not f.startswith('.')]

        if not txt_files:
            print("错误: 没有找到txt文件")
            return

        print(f"找到 {len(txt_files)} 个txt文件")
        print(f"图像尺寸: {self.img_width}x{self.img_height}")
        print(f"类别映射: {self.id_to_class}")
        print("开始转换...\n")

        success_count = 0
        for i, txt_file in enumerate(txt_files, 1):
            print(f"[{i:3d}/{len(txt_files)}] 处理: {txt_file}", end="")
            if self.convert_file(txt_file):
                print(" ✓")
                success_count += 1
            else:
                print(" ✗")

        # 生成统计信息
        print(f"\n{'=' * 50}")
        print("转换完成!")
        print(f"总计: {len(txt_files)} 个文件")
        print(f"成功: {success_count} 个")
        print(f"失败: {len(txt_files) - success_count} 个")
        print(f"输出目录: {os.path.abspath(self.output_folder)}")
        print(f"{'=' * 50}")


def load_class_mapping(json_path):
    """
    从JSON文件加载类别映射

    Args:
        json_path (str): JSON文件路径

    Returns:
        dict: 类别映射字典
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            class_mapping = json.load(f)
        print(f"成功加载类别映射: {class_mapping}")
        return class_mapping
    except Exception as e:
        print(f"错误: 无法加载类别映射文件 {json_path}: {e}")
        print("使用默认类别映射...")
        # 如果加载失败，使用你提供的映射
        return {
            "U-type": 1,
            "inverted-U": 2,
            "chain": 3,
            "special": 4
        }


def main():
    """
    主函数
    """
    # ========== 配置参数 ==========

    # 类别映射文件路径（可选）
    CLASS_JSON_PATH = "cluster_datasets_classes.json"  # 类别映射JSON文件

    # 文件夹路径（相对路径）
    IMAGE_FOLDER = "./pic"  # 图像文件夹（.bmp文件）
    TXT_FOLDER = "./labels_v10_four"  # YOLO标注文件夹（.txt文件）
    OUTPUT_FOLDER = "./annotations"  # 输出的XML文件夹

    # 数据集信息
    DATASET_NAME = "ClusterDataSet"
    IMG_WIDTH = 1280
    IMG_HEIGHT = 800

    # ========== 参数设置结束 ==========

    # 加载类别映射
    if os.path.exists(CLASS_JSON_PATH):
        class_mapping = load_class_mapping(CLASS_JSON_PATH)
    else:
        print(f"注意: 未找到类别映射文件 {CLASS_JSON_PATH}")
        print("使用内置类别映射...")
        class_mapping = {
            "U-type": 1,
            "inverted-U": 2,
            "chain": 3,
            "special": 4
        }

    # 检查文件夹是否存在
    for folder in [IMAGE_FOLDER, TXT_FOLDER]:
        if not os.path.exists(folder):
            print(f"警告: 文件夹不存在 {folder}")

    # 创建转换器
    converter = YOLOToVOCConverter(
        class_mapping=class_mapping,
        image_folder=IMAGE_FOLDER,
        txt_folder=TXT_FOLDER,
        output_folder=OUTPUT_FOLDER,
        dataset_name=DATASET_NAME,
        img_width=IMG_WIDTH,
        img_height=IMG_HEIGHT
    )

    # 执行转换
    converter.convert_all()


if __name__ == "__main__":
    main()