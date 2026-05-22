# 利用k-fold进行不同的训练集验证集划分

import os
import yaml
import shutil
from sklearn.model_selection import KFold
import random


def create_kfold_dataset(original_dataset_path, kfold_dataset_path):
    """创建5折交叉验证的数据集和YAML文件"""
    # 创建输出目录
    if not os.path.exists(kfold_dataset_path):
        os.makedirs(kfold_dataset_path)
    # 读取原始数据集结构
    original_images_train = os.path.join(original_dataset_path, "images", "train")
    original_images_val = os.path.join(original_dataset_path, "images", "val")
    original_labels_train = os.path.join(original_dataset_path, "labels", "train")
    original_labels_val = os.path.join(original_dataset_path, "labels", "val")
    # 获取所有训练图像文件（合并训练集和验证集，用于K折划分）
    train_image_files = []
    # 从训练集获取文件
    if os.path.exists(original_images_train):
        for file in os.listdir(original_images_train):
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                train_image_files.append(('train', file))
    # 从验证集获取文件
    if os.path.exists(original_images_val):
        for file in os.listdir(original_images_val):
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                train_image_files.append(('val', file))

    print(f"找到 {len(train_image_files)} 个图像文件 (训练集+验证集)")

    # 随机打乱文件列表
    random.seed(42)
    random.shuffle(train_image_files)
    # 创建K折划分
    k_folds = 5
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    # 为每一折创建数据集
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_image_files)):
        print(f"\n正在创建第 {fold + 1} 折数据集...")
        # 创建折的目录结构
        fold_path = os.path.join(kfold_dataset_path, f"fold_{fold + 1}")
        fold_images_train = os.path.join(fold_path, "images", "train")
        fold_images_val = os.path.join(fold_path, "images", "val")
        fold_labels_train = os.path.join(fold_path, "labels", "train")
        fold_labels_val = os.path.join(fold_path, "labels", "val")
        # 创建目录
        os.makedirs(fold_images_train, exist_ok=True)
        os.makedirs(fold_images_val, exist_ok=True)
        os.makedirs(fold_labels_train, exist_ok=True)
        os.makedirs(fold_labels_val, exist_ok=True)
        # 获取当前折的训练集和验证集文件
        fold_train_files = [train_image_files[i] for i in train_idx]
        fold_val_files = [train_image_files[i] for i in val_idx]
        print(f"第 {fold + 1} 折 - 训练集: {len(fold_train_files)} 个文件, 验证集: {len(fold_val_files)} 个文件")
        # 复制训练集文件
        for source_type, file in fold_train_files:
            # 确定源路径
            if source_type == 'train':
                src_image_dir = original_images_train
                src_label_dir = original_labels_train
            else:  # 'val'
                src_image_dir = original_images_val
                src_label_dir = original_labels_val
            # 复制图像文件
            src_image = os.path.join(src_image_dir, file)
            dst_image = os.path.join(fold_images_train, file)
            if os.path.exists(src_image):
                shutil.copy2(src_image, dst_image)
            else:
                print(f"警告: 图像文件不存在: {src_image}")
                continue

            # 复制标签文件
            label_file = os.path.splitext(file)[0] + '.txt'
            src_label = os.path.join(src_label_dir, label_file)
            dst_label = os.path.join(fold_labels_train, label_file)
            if os.path.exists(src_label):
                shutil.copy2(src_label, dst_label)
            else:
                print(f"警告: 标签文件不存在: {src_label}")

        # 复制验证集文件
        for source_type, file in fold_val_files:
            # 确定源路径
            if source_type == 'train':
                src_image_dir = original_images_train
                src_label_dir = original_labels_train
            else:  # 'val'
                src_image_dir = original_images_val
                src_label_dir = original_labels_val
            # 复制图像文件
            src_image = os.path.join(src_image_dir, file)
            dst_image = os.path.join(fold_images_val, file)
            if os.path.exists(src_image):
                shutil.copy2(src_image, dst_image)
            else:
                print(f"警告: 图像文件不存在: {src_image}")
                continue
            # 复制标签文件
            label_file = os.path.splitext(file)[0] + '.txt'
            src_label = os.path.join(src_label_dir, label_file)
            dst_label = os.path.join(fold_labels_val, label_file)
            if os.path.exists(src_label):
                shutil.copy2(src_label, dst_label)
            else:
                print(f"警告: 标签文件不存在: {src_label}")

        # 创建YAML配置文件
        create_yaml_config(fold_path, fold + 1)
        print(f"第 {fold + 1} 折数据集创建完成: {fold_path}")


def create_yaml_config(fold_path, fold_num):
    """为每一折创建YAML配置文件"""
    config = {
        'path': fold_path,
        'train': 'images/train',
        'val': 'images/val',
        'test': '',  # 可以留空或指向测试集
        'nc': 4,
        'names': {
            0: 'U-type',
            1: 'inverted-U',
            2: 'chain',
            3: 'special'
        }
    }
    yaml_path = os.path.join(fold_path, f"dataset_fold_{fold_num}.yaml")
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    print(f"YAML配置文件已创建: {yaml_path}")


def create_combined_yaml_config(kfold_dataset_path):
    """创建合并的YAML配置文件，用于训练所有折"""
    config = {
        'path': kfold_dataset_path,
        'k_folds': 5,
        'nc': 4,
        'names': {
            0: 'U-shape',
            1: 'inverted-U',
            2: 'long-chain',
            3: 'special'
        },
        'folds': {}
    }
    for fold in range(1, 6):
        fold_path = os.path.join(kfold_dataset_path, f"fold_{fold}")
        config['folds'][f'fold_{fold}'] = {
            'train': os.path.join(fold_path, "images", "train"),
            'val': os.path.join(fold_path, "images", "val"),
            'config': os.path.join(fold_path, f"dataset_fold_{fold}.yaml")
        }
    combined_yaml_path = os.path.join(kfold_dataset_path, "kfold_dataset_config.yaml")
    with open(combined_yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    print(f"合并配置文件已创建: {combined_yaml_path}")


def verify_dataset(kfold_dataset_path):
    """验证生成的数据集"""

    print("\n验证数据集...")
    for fold in range(1, 6):
        fold_path = os.path.join(kfold_dataset_path, f"fold_{fold}")
        images_train = os.path.join(fold_path, "images", "train")
        images_val = os.path.join(fold_path, "images", "val")
        labels_train = os.path.join(fold_path, "labels", "train")
        labels_val = os.path.join(fold_path, "labels", "val")
        train_images_count = len(
            [f for f in os.listdir(images_train) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
        val_images_count = len(
            [f for f in os.listdir(images_val) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
        train_labels_count = len([f for f in os.listdir(labels_train) if f.endswith('.txt')])
        val_labels_count = len([f for f in os.listdir(labels_val) if f.endswith('.txt')])

        print(f"第 {fold} 折:")
        print(f"  训练集: {train_images_count} 图像, {train_labels_count} 标签")
        print(f"  验证集: {val_images_count} 图像, {val_labels_count} 标签")
        # 检查图像和标签是否匹配
        if train_images_count != train_labels_count:
            print(f"  警告: 训练集图像和标签数量不匹配!")
        if val_images_count != val_labels_count:
            print(f"  警告: 验证集图像和标签数量不匹配!")


def generate_dataset_summary(kfold_dataset_path, original_dataset_path):
    """生成数据集摘要报告"""
    print("\n生成数据集摘要报告...")

    # 统计原始数据集
    original_train_images = len([f for f in os.listdir(os.path.join(original_dataset_path, "images", "train"))
                                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
    original_val_images = len([f for f in os.listdir(os.path.join(original_dataset_path, "images", "val"))
                               if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
    # 统计每折数据集情况
    fold_statistics = []
    class_names = {0: 'U-type', 1: 'inverted-U', 2: 'chain', 3: 'special'}
    for fold in range(1, 6):
        fold_path = os.path.join(kfold_dataset_path, f"fold_{fold}")

        # 统计图像数量
        train_images_count = len([f for f in os.listdir(os.path.join(fold_path, "images", "train"))
                                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
        val_images_count = len([f for f in os.listdir(os.path.join(fold_path, "images", "val"))
                                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
        # 统计各类别数量
        train_class_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        val_class_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        # 统计训练集类别
        train_labels_dir = os.path.join(fold_path, "labels", "train")
        if os.path.exists(train_labels_dir):
            for label_file in os.listdir(train_labels_dir):
                if label_file.endswith('.txt'):
                    label_path = os.path.join(train_labels_dir, label_file)
                    try:
                        with open(label_path, 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                            for line in lines:
                                line = line.strip()
                                if line:
                                    class_id = int(line.split()[0])
                                    if class_id in train_class_counts:
                                        train_class_counts[class_id] += 1
                    except Exception as e:
                        print(f"读取标签文件错误 {label_path}: {e}")
        # 统计验证集类别
        val_labels_dir = os.path.join(fold_path, "labels", "val")
        if os.path.exists(val_labels_dir):
            for label_file in os.listdir(val_labels_dir):
                if label_file.endswith('.txt'):
                    label_path = os.path.join(val_labels_dir, label_file)
                    try:
                        with open(label_path, 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                            for line in lines:
                                line = line.strip()
                                if line:
                                    class_id = int(line.split()[0])
                                    if class_id in val_class_counts:
                                        val_class_counts[class_id] += 1
                    except Exception as e:
                        print(f"读取标签文件错误 {label_path}: {e}")
        # 计算总数和比例
        train_total_objects = sum(train_class_counts.values())
        val_total_objects = sum(val_class_counts.values())
        train_class_ratios = {}
        val_class_ratios = {}
        for class_id in range(4):
            train_class_ratios[class_id] = (
                        train_class_counts[class_id] / train_total_objects * 100) if train_total_objects > 0 else 0
            val_class_ratios[class_id] = (
                        val_class_counts[class_id] / val_total_objects * 100) if val_total_objects > 0 else 0
        fold_statistics.append({
            'fold': fold,
            'train_images': train_images_count,
            'val_images': val_images_count,
            'train_class_counts': train_class_counts.copy(),
            'val_class_counts': val_class_counts.copy(),
            'train_total_objects': train_total_objects,
            'val_total_objects': val_total_objects,
            'train_class_ratios': train_class_ratios.copy(),
            'val_class_ratios': val_class_ratios.copy()
        })

    # 生成详细的统计报告
    summary = f"""
数据集交叉验证摘要报告
{'=' * 50}

原始数据集信息:
- 路径: {original_dataset_path}
- 训练集图像: {original_train_images}
- 验证集图像: {original_val_images}
- 总计图像: {original_train_images + original_val_images}

5折交叉验证数据集:
- 路径: {kfold_dataset_path}
- 总折数: 5

各折数据集详细统计:
{'=' * 50}
"""

    for stats in fold_statistics:
        summary += f"""
第 {stats['fold']} 折:
  图像数量:
    - 训练集: {stats['train_images']} 张图像
    - 验证集: {stats['val_images']} 张图像
    - 总计: {stats['train_images'] + stats['val_images']} 张图像

  训练集目标检测统计:
    - 总目标数: {stats['train_total_objects']}
    - 各类别数量及比例:"""

        for class_id in range(4):
            summary += f"\n      {class_names[class_id]}: {stats['train_class_counts'][class_id]} 个 ({stats['train_class_ratios'][class_id]:.1f}%)"

        summary += f"""

  验证集目标检测统计:
    - 总目标数: {stats['val_total_objects']}
    - 各类别数量及比例:"""

        for class_id in range(4):
            summary += f"\n      {class_names[class_id]}: {stats['val_class_counts'][class_id]} 个 ({stats['val_class_ratios'][class_id]:.1f}%)"

        summary += f"""

  训练集/验证集比例:
    - 图像数量: {stats['train_images']}:{stats['val_images']} ({stats['train_images'] / (stats['train_images'] + stats['val_images']) * 100:.1f}% / {stats['val_images'] / (stats['train_images'] + stats['val_images']) * 100:.1f}%)
    - 目标数量: {stats['train_total_objects']}:{stats['val_total_objects']} ({stats['train_total_objects'] / (stats['train_total_objects'] + stats['val_total_objects']) * 100:.1f}% / {stats['val_total_objects'] / (stats['train_total_objects'] + stats['val_total_objects']) * 100:.1f}%)
{'-' * 50}"""

    # 添加总体统计
    total_train_images = sum([stats['train_images'] for stats in fold_statistics])
    total_val_images = sum([stats['val_images'] for stats in fold_statistics])
    total_train_objects = sum([stats['train_total_objects'] for stats in fold_statistics])
    total_val_objects = sum([stats['val_total_objects'] for stats in fold_statistics])

    # 计算总体类别分布
    overall_train_class_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    overall_val_class_counts = {0: 0, 1: 0, 2: 0, 3: 0}

    for stats in fold_statistics:
        for class_id in range(4):
            overall_train_class_counts[class_id] += stats['train_class_counts'][class_id]
            overall_val_class_counts[class_id] += stats['val_class_counts'][class_id]

    summary += f"""

总体统计:
{'=' * 50}
总图像数量:
- 训练集: {total_train_images} 张 (平均每折: {total_train_images / 5:.1f} 张)
- 验证集: {total_val_images} 张 (平均每折: {total_val_images / 5:.1f} 张)
- 总计: {total_train_images + total_val_images} 张

总目标检测数量:
- 训练集: {total_train_objects} 个目标 (平均每折: {total_train_objects / 5:.1f} 个)
- 验证集: {total_val_objects} 个目标 (平均每折: {total_val_objects / 5:.1f} 个)
- 总计: {total_train_objects + total_val_objects} 个目标

总体类别分布 (训练集):
"""

    for class_id in range(4):
        ratio = (overall_train_class_counts[class_id] / total_train_objects * 100) if total_train_objects > 0 else 0
        summary += f"- {class_names[class_id]}: {overall_train_class_counts[class_id]} 个 ({ratio:.1f}%)\n"

    summary += """
总体类别分布 (验证集):
"""

    for class_id in range(4):
        ratio = (overall_val_class_counts[class_id] / total_val_objects * 100) if total_val_objects > 0 else 0
        summary += f"- {class_names[class_id]}: {overall_val_class_counts[class_id]} 个 ({ratio:.1f}%)\n"

    summary += f"""
生成的目录结构:
{kfold_dataset_path}/
├── fold_1/
│   ├── images/train/     # 第1折训练图像
│   ├── images/val/       # 第1折验证图像
│   ├── labels/train/     # 第1折训练标签
│   ├── labels/val/       # 第1折验证标签
│   └── dataset_fold_1.yaml
├── fold_2/
│   └── ...
├── ...
├── kfold_dataset_config.yaml
└── train_kfold_models.py

"""

    summary_path = os.path.join(kfold_dataset_path, "dataset_summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    print(f"数据集摘要报告已生成: {summary_path}")
    # 在控制台输出简化版报告
    print(f"\n数据集交叉验证摘要 (简化版):")
    print(f"总图像: {original_train_images + original_val_images} 张")
    print(f"总折数: 5 折")
    for stats in fold_statistics:
        print(f"第 {stats['fold']} 折: 训练集 {stats['train_images']} 张, 验证集 {stats['val_images']} 张")
    return summary


if __name__ == '__main__':
    print("开始创建5折交叉验证数据集...")
    original_dataset_path = r"D:\project_SJ\ultralytics-main\datasets\dataset-v10-four-augment"     # TODO
    kfold_dataset_path = r"D:\project_SJ\ultralytics-main\datasets\dataset-v10-four-kfold-augment"    # TODO
    # 创建K折数据集
    create_kfold_dataset(original_dataset_path, kfold_dataset_path)
    # 创建合并配置文件
    create_combined_yaml_config(kfold_dataset_path)
    # 验证数据集
    verify_dataset(kfold_dataset_path)
    # 生成摘要报告
    generate_dataset_summary(kfold_dataset_path, original_dataset_path)
    print("\n5折交叉验证数据集创建完成!")
    print(f"数据集位置: {kfold_dataset_path}")

