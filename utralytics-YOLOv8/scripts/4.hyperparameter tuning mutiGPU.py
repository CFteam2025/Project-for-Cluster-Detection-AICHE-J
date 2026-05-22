import os
import itertools
import glob
import csv
import pandas as pd
import time
from datetime import datetime
import torch
import traceback
from ultralytics import YOLO


def hyperparameter_tuning_single_fold():
    """超参数调优 - 单次训练，给定参数范围"""
    # 定义超参数范围
    hyperparameter_ranges = {
        'lr0': [0.01],
        'lrf': [0.01],
        'optimizer': ['SGD'],
        'cos_lr': [False],
        'momentum': [0.937],
        'weight_decay': [0.0005],
        'box': [6.5],
        'cls': [0.75],
        'dfl': [1],
        'close_mosaic': [30]
    }

    # 生成参数组合
    param_combinations = generate_param_combinations(hyperparameter_ranges, max_combinations=25)
    print(f"开始超参数调优，共 {len(param_combinations)} 种组合")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 存储所有实验结果
    all_results = []
    config_path = r"D:\project_SJ\ultralytics-main\datasets\dataset-v10-four-kfold\fold_4\dataset_fold_4.yaml"

    for i, params in enumerate(param_combinations, 1):
        print(f"\n{'=' * 60}")
        print(f"训练组合 {i}/{len(param_combinations)}: {params['name']}")
        print(params)
        print(f"{'=' * 60}")

        try:
            # 初始化模型
            model = YOLO(r"D:/project_SJ/ultralytics-main/ultralytics/cfg/models/v8/yolov8m.yaml")

            # 创建唯一的实验名称，包含时间戳
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            exp_name = f"SJ_tuning/close_mosaic_1/{timestamp}_{params['name']}"

            # 训练参数 - 多GPU配置
            train_kwargs = {
                'epochs': 120,  # 先测试100个epoch
                'batch': 24,
                'device': [0, 1],
                'data': config_path,
                'name': exp_name,  # 使用唯一名称
                'cos_lr': params['cos_lr'],
                'lr0': params['lr0'],
                'lrf': params['lrf'],
                'optimizer': params['optimizer'],
                'momentum': params['momentum'],
                'weight_decay': params['weight_decay'],
                'box': params['box'],
                'cls': params['cls'],
                'dfl': params['dfl'],
                'close_mosaic': params['close_mosaic'],
                'single_cls': False,
                'patience': 200,
                'workers': 4,
                'cache': False,
                'save_json': True,  # 保存JSON结果
            }

            print(f"开始训练，实验名称: {exp_name}")
            start_time = time.time()
            # 训练模型（多GPU会返回None）
            model.train(**train_kwargs)
            train_time = time.time() - start_time
            print(f"训练完成，耗时: {train_time:.1f}秒")

            # 等待文件写入完成
            time.sleep(2)
            # 从文件系统读取最佳结果
            result_info = extract_results_from_files(exp_name, params, train_time)

            if result_info:
                all_results.append(result_info)
                print(f"✅ 完成: {params['name']}")
                print(f"   最佳mAP50: {result_info['best_map50']:.4f} (epoch {result_info['best_epoch']})")
                print(f"   最终mAP50: {result_info['final_map50']:.4f}")
                print(f"   学习率: {params['lr0']}, {params['lrf']}, cos_lr: {params['cos_lr']}")
                print(f"   训练时间: {train_time:.1f}秒")
            else:
                print(f"⚠️  警告: {params['name']} - 未能提取有效指标")

        except Exception as e:
            print(f"❌ 失败: {params['name']} - {e}")
            traceback.print_exc()
            continue

    # 保存结果到文件
    if all_results:
        save_results_to_txt(all_results)
    else:
        print("没有有效结果可保存")

    return all_results


def extract_results_from_files(exp_name, params, train_time):
    """
    从文件系统读取训练结果，获取最佳指标
    多GPU训练时，结果保存在runs/detect/目录下
    """
    try:
        # 1. 查找实验目录
        exp_dir = find_experiment_dir(exp_name)
        if not exp_dir:
            print(f"未找到实验目录: {exp_name}")
            return None
        print(f"找到实验目录: {exp_dir}")

        # 2. 读取CSV结果文件
        csv_path = os.path.join(exp_dir, "results.csv")

        # 3. 读取并分析CSV数据
        df = pd.read_csv(csv_path)
        if len(df) == 0:
            print(f"CSV文件为空: {csv_path}")
            return None

        # 4. 找到最佳指标（不是最后指标！）
        best_map50 = 0
        best_map = 0
        best_precision = 0
        best_recall = 0
        best_epoch = 0

        # 先清理列名，去掉多余的空格
        df.columns = df.columns.str.strip()
        print(f"清理后列名: {list(df.columns)}")

        if 'metrics/mAP50(B)' in df.columns:
            # 确保数据类型是数值
            df['metrics/mAP50(B)'] = pd.to_numeric(df['metrics/mAP50(B)'], errors='coerce')

            # 找到mAP50最高的行（忽略NaN）
            if df['metrics/mAP50(B)'].notna().any():
                best_idx = df['metrics/mAP50(B)'].idxmax()
                best_row = df.iloc[best_idx]

                # 正确赋值：mAP50(B)对应mAP50，mAP50-95(B)对应mAP
                best_map50 = float(best_row['metrics/mAP50(B)'])

                if 'epoch' in df.columns:
                    best_epoch = int(best_row['epoch'])
                else:
                    best_epoch = best_idx + 1

                if 'metrics/mAP50-95(B)' in df.columns:
                    best_map = float(best_row['metrics/mAP50-95(B)'])
                if 'metrics/precision(B)' in df.columns:
                    best_precision = float(best_row['metrics/precision(B)'])
                if 'metrics/recall(B)' in df.columns:
                    best_recall = float(best_row['metrics/recall(B)'])

                print(f"✓ 找到最佳结果: epoch={best_epoch}, mAP50={best_map50:.4f}, mAP50-95={best_map:.4f}")
            else:
                print("⚠️ mAP50列所有值都是NaN")
        else:
            print(f"❌ 找不到metrics/mAP50(B)列")
            print(f"可用列: {list(df.columns)}")

        # 5. 获取最终指标（最后一行）
        if 'metrics/mAP50(B)' in df.columns:
            final_row = df.iloc[-1]
            final_map50 = float(final_row['metrics/mAP50(B)'])
            if 'metrics/mAP50-95(B)' in df.columns:
                final_map = float(final_row['metrics/mAP50-95(B)'])
            else:
                final_map = 0
        else:
            final_map50 = 0
            final_map = 0

        print(f"最终结果: epoch={len(df)}, mAP50={final_map50:.4f}, mAP50-95={final_map:.4f}")

        # 6. 获取fitness（如果有）
        fitness = float(df.iloc[best_idx]['fitness']) if 'fitness' in df.columns else 0

        # 7. 获取最佳模型路径
        best_model_path = os.path.join(exp_dir, "weights", "best.pt")
        best_model_size = 0
        if best_model_path and os.path.exists(best_model_path):
            best_model_size = os.path.getsize(best_model_path) / (1024 * 1024)  # MB

        # 8. 构建结果字典
        result_info = {
            'param_name': params['name'],
            'exp_name': exp_name,
            'exp_dir': exp_dir,
            'lr0': params['lr0'],
            'lrf': params['lrf'],
            'cos_lr': params['cos_lr'],
            'momentum': params['momentum'],
            'weight_decay': params['weight_decay'],
            'optimizer': params['optimizer'],
            'box': params['box'],
            'cls': params['cls'],
            'dfl': params['dfl'],

            # 最佳指标
            'best_map50': best_map50,
            'best_map': best_map,
            'best_precision': best_precision,
            'best_recall': best_recall,
            'best_fitness': fitness,
            'best_epoch': best_epoch,

            # 最终指标
            'final_map50': final_map50,
            'final_map': final_map,

            # 其他信息
            'train_time': train_time,
            'best_model_path': best_model_path,
            'best_model_size_mb': best_model_size,
            'total_epochs': len(df),
        }

        # 9. 打印最佳结果信息
        print(f"最佳结果: epoch {best_epoch}/{len(df)}, mAP50={best_map50:.4f}")

        return result_info

    except Exception as e:
        print(f"从文件提取结果时出错: {e}")
        traceback.print_exc()
        return None


def find_experiment_dir(exp_name):
    """查找实验目录"""
    # 先尝试精确匹配
    exp_dir = os.path.join("runs", "detect", exp_name)
    if os.path.exists(exp_dir):
        return exp_dir
    return None


def generate_param_combinations(param_ranges, max_combinations=100):
    """生成参数组合，限制总数"""
    combinations = []
    # 基础参数组合
    base_combinations = list(itertools.product(
        param_ranges['cos_lr'],
        param_ranges['lr0'],
        param_ranges['lrf'],
        param_ranges['optimizer'],
        param_ranges['momentum'],
        param_ranges['weight_decay'],
        param_ranges['box'],
        param_ranges['cls'],
        param_ranges['dfl'],
        param_ranges['close_mosaic']
    ))

    # 限制组合数量
    selected_combinations = base_combinations[:max_combinations]
    for i, (cos_lr, lr0, lrf, optimizer, momentum, weight_decay, box, cls, dfl, close_mosaic) in enumerate(selected_combinations):
        param_set = {
            'name': f"set_{i + 1:02d}",
            'cos_lr': cos_lr,
            'lr0': lr0,
            'lrf': lrf,
            'optimizer': optimizer,
            'momentum': momentum,
            'weight_decay': weight_decay,
            'box': box,
            'cls': cls,
            'dfl': dfl,
            'close_mosaic': close_mosaic
        }
        combinations.append(param_set)

    print(f"生成 {len(combinations)} 个参数组合")
    return combinations


def save_results_to_txt(all_results):
    """保存结果到txt文件"""
    if not all_results:
        print("没有有效结果可保存")
        return None

    # 按最佳mAP50排序
    all_results.sort(key=lambda x: x['best_map50'], reverse=True)

    # 创建结果目录
    results_dir = "hyperparameter_tuning_results"
    os.makedirs(results_dir, exist_ok=True)

    # 生成文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_path = os.path.join(results_dir, f"hyperparameter_tuning_results_{timestamp}.txt")

    with open(txt_path, 'w', encoding='utf-8') as f:
        # 写入标题
        f.write("超参数调优实验结果报告\n")
        f.write("=" * 80 + "\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总实验数量: {len(all_results)}\n")
        f.write("注意: 显示的是最佳epoch的指标，不是最终指标\n")
        f.write("=" * 80 + "\n\n")

        # 写入最佳结果
        f.write("🎯 最佳参数组合:\n")
        f.write("-" * 80 + "\n")
        best_result = all_results[0]
        f.write(f"参数名称: {best_result['param_name']}\n")
        f.write(f"实验名称: {best_result['exp_name']}\n")
        f.write(f"最佳mAP50:     {best_result['best_map50']:.4f} (epoch {best_result['best_epoch']})\n")
        f.write(f"最佳mAP50-95:  {best_result['best_map']:.4f}\n")
        f.write(f"最终mAP50:     {best_result['final_map50']:.4f}\n")
        f.write(f"最佳Precision: {best_result['best_precision']:.4f}\n")
        f.write(f"最佳Recall:    {best_result['best_recall']:.4f}\n")
        f.write(f"Fitness:       {best_result['best_fitness']:.4f}\n")
        f.write(f"Cosine_LR:     {best_result['cos_lr']}\n")
        f.write(f"初始学习率:    {best_result['lr0']}\n")
        f.write(f"最终学习率因子: {best_result['lrf']}\n")
        f.write(f"优化器:        {best_result['optimizer']}\n")
        f.write(f"Momentum:      {best_result['momentum']}\n")
        f.write(f"Weight_decay:  {best_result['weight_decay']}\n")
        f.write(f"box损失权重:   {best_result['box']}\n")
        f.write(f"cls损失权重:   {best_result['cls']}\n")
        f.write(f"dfl损失权重:   {best_result['dfl']}\n")
        f.write(f"训练时间:      {best_result['train_time']:.1f}秒\n")
        if best_result['best_model_path']:
            f.write(f"最佳模型:      {best_result['best_model_path']} ({best_result['best_model_size_mb']:.1f}MB)\n")
        f.write("\n")

        # 写入所有结果排名
        f.write("📊 所有参数组合排名（按最佳mAP50）:\n")
        f.write("-" * 80 + "\n")
        f.write(
            f"{'排名':<4} {'参数名':<12} {'最佳mAP50':<10} {'最佳epoch':<10} {'最终mAP50':<10} {'优化器':<8} {'学习率':<12}\n")
        f.write("-" * 80 + "\n")

        for i, result in enumerate(all_results, 1):
            f.write(f"{i:<4} {result['param_name']:<12} {result['best_map50']:.4f}     "
                    f"{result['best_epoch']:<10} {result['final_map50']:.4f}     "
                    f"{result['optimizer']:<8} {result['lr0']}/{result['lrf']:<12}\n")
        f.write("\n")


        # 写入训练统计
        f.write("\n📈 训练统计:\n")
        f.write("-" * 80 + "\n")
        # 计算平均训练时间
        avg_time = sum(r['train_time'] for r in all_results) / len(all_results)
        f.write(f"平均训练时间: {avg_time:.1f}秒\n")

        # 计算最佳指标与最终指标的差异
        map50_diffs = [r['best_map50'] - r['final_map50'] for r in all_results]
        avg_diff = sum(map50_diffs) / len(map50_diffs)
        f.write(f"平均mAP50下降 (最佳→最终): {avg_diff:.4f}\n")

        # 查找过拟合最严重的实验
        if map50_diffs:
            max_diff_idx = map50_diffs.index(max(map50_diffs))
            worst_result = all_results[max_diff_idx]
            f.write(f"过拟合最严重: {worst_result['param_name']}, "
                    f"下降 {max(map50_diffs):.4f} "
                    f"(最佳{worst_result['best_map50']:.4f}→最终{worst_result['final_map50']:.4f})\n")

        # 写入建议
        f.write("\n💡 调优建议:\n")
        f.write("-" * 80 + "\n")
        f.write("1. 使用排名前3的参数组合进行最终训练（400 epochs）\n")
        f.write("2. 注意观察过拟合情况，最佳epoch与最终epoch的指标差异\n")
        f.write("3. 如果时间允许，可以在最佳参数附近进行更精细的搜索\n")
        f.write("4. 学习率对模型性能影响很大，lr0和lrf都需要仔细调优\n")
        f.write("5. 多GPU训练时注意使用SyncBatchNorm保持稳定性\n")
        f.write("6. 考虑使用早停（patience）防止过拟合\n")
        f.write("7. 保存最佳模型用于后续推理\n")

    print(f"\n✅ 结果已保存到: {txt_path}")
    return txt_path


if __name__ == '__main__':
    # 设置多GPU环境变量
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

    print("开始超参数调优实验...")
    print(f"使用GPU: {torch.cuda.device_count()} 个")
    print("注意: 多GPU训练时，结果从文件系统读取最佳指标")

    results = hyperparameter_tuning_single_fold()

    if results:
        print(f"\n🎉 调优完成！共完成 {len(results)} 个实验")
        best_result = max(results, key=lambda x: x['best_map50'])
        print(f"🏆 最佳mAP50: {best_result['best_map50']:.4f} "
              f"(epoch {best_result['best_epoch']}, 参数: {best_result['param_name']})")
        print(f"📁 实验目录: {best_result['exp_dir']}")
    else:
        print("\n❌ 调优失败，没有有效结果")