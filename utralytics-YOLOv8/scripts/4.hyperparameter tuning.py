import os
import itertools
from ultralytics import YOLO
from datetime import datetime


def hyperparameter_tuning_single_fold():
    """超参数调优 - 单次训练，给定参数范围"""
    # 定义超参数范围
    hyperparameter_ranges = {

        'lr0': [0.01],
        'lrf': [0.01,0.05,0.1],
        'optimizer': ['Adam'],
        'cos_lr': [False],
        'momentum': [0.937],
        'weight_decay': [0.0005],
        'box': [7.5],
        'cls': [0.5],
        'dfl': [1.5]
    }

    # 生成参数组合（限制数量，避免太多组合）
    param_combinations = generate_param_combinations(hyperparameter_ranges, max_combinations=9)
    print(f"开始超参数调优，共 {len(param_combinations)} 种组合")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    # 存储所有实验结果
    all_results = []
    config_path = r"D:\project_SJ\ultralytics-main\datasets\dataset-v10-four-kfold\fold_1\dataset_fold_1.yaml"    # 数据位置

    for i, params in enumerate(param_combinations, 1):
        print(f"\n{'=' * 60}")
        print(f"训练组合 {i}/{len(param_combinations)}: {params['name']}")
        print(params)
        print(f"{'=' * 60}")
        try:
            # 初始化模型
            model = YOLO(
                r"D:/project_SJ/ultralytics-main/ultralytics/cfg/models/v8/yolov8m.yaml")
            # 训练参数
            train_kwargs = {
                'epochs': 3,
                'batch': 24,
                'device': [0,1],
                'data': config_path,
                'name': f"tuning.v10.SGD.loss11.7.3.fold1_{params['name']}",
                'cos_lr': params['cos_lr'],
                'lr0': params['lr0'],
                'lrf': params['lrf'],
                'optimizer': params['optimizer'],
                'momentum': params['momentum'],
                'weight_decay': params['weight_decay'],
                'box': params['box'],
                'cls': params['cls'],
                'dfl': params['dfl'],
                'single_cls': False,
                'patience': 50,
                'cache': True,
                'workers': 0,
                'pretrained': True,
            }
            # 训练模型
            results = model.train(**train_kwargs)
            # 兼容新旧版本的指标获取
            if hasattr(results, 'metrics'):
                metrics = results.metrics
            elif hasattr(results, 'results_dict'):
                metrics = results.results_dict
            else:
                print(f"❌ 失败: {params['name']} - 未找到指标数据")
                continue
            # 获取评估指标
            # metrics = results.results_dict
            result_info = {
                'param_name': params['name'],
                'lr0': params['lr0'],
                'lrf': params['lrf'],
                'cos_lr': params['cos_lr'],
                'momentum': params['momentum'],
                'weight_decay': params['weight_decay'],
                'optimizer': params['optimizer'],
                'box': params['box'],
                'cls': params['cls'],
                'dfl': params['dfl'],
                'map50': metrics.get('metrics/mAP50(B)', 0),
                'map': metrics.get('metrics/mAP50-95(B)', 0),
                'precision': metrics.get('metrics/precision(B)', 0),
                'recall': metrics.get('metrics/recall(B)', 0),
                'fitness': results.fitness if hasattr(results, 'fitness') else 0,
                'train_time': sum(results.times.values()) if (
                            hasattr(results, 'times') and isinstance(results.times, dict)) else 0
            }
                #'fitness': results.fitness,
                #'train_time': sum(results.times.values()) if hasattr(results, 'times') else 0
            all_results.append(result_info)
            print(f"✅ 完成: {params['name']}")
            print(f"   mAP50: {result_info['map50']:.4f}, mAP50-95: {result_info['map']:.4f}")
            print(f"   学习率: {params['lr0'], params['lrf'],params['cos_lr']}, "
                  f"   优化器: {params['optimizer'], params['momentum'], params['optimizer']}"
                  f"   损失权重: {params['box'], params['cls'], params['dfl']}")

        except Exception as e:
            print(f"❌ 失败: {params['name']} - {e}")
            continue

    # 保存结果到文件
    save_results_to_txt(all_results)
    return all_results


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
        param_ranges['dfl']
    ))
    # 限制组合数量
    selected_combinations = base_combinations[:max_combinations]
    for i, (cos_lr, lr0, lrf, optimizer, momentum, weight_decay, box, cls, dfl) in enumerate(selected_combinations):
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
            'dfl':dfl
        }
        combinations.append(param_set)
    print(combinations)
    return combinations


def save_results_to_txt(all_results):
    """保存结果到txt文件"""
    if not all_results:
        print("没有有效结果可保存")
        return

    # 按mAP50排序
    all_results.sort(key=lambda x: x['map50'], reverse=True)
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
        f.write("=" * 80 + "\n\n")

        # 写入最佳结果
        f.write("🎯 最佳参数组合:\n")
        f.write("-" * 80 + "\n")
        best_result = all_results[0]
        f.write(f"参数名称: {best_result['param_name']}\n")
        f.write(f"mAP50:     {best_result['map50']:.4f}\n")
        f.write(f"mAP50-95:  {best_result['map']:.4f}\n")
        f.write(f"Precision: {best_result['precision']:.4f}\n")
        f.write(f"Recall:    {best_result['recall']:.4f}\n")
        f.write(f"Fitness:   {best_result['fitness']:.4f}\n")
        f.write(f"Cosine_LR: {best_result['cos_lr']}\n")
        f.write(f"初始学习率:    {best_result['lr0']}\n")
        f.write(f"最终学习率因子: {best_result['lrf']}\n")
        f.write(f"优化器:    {best_result['optimizer']}\n")
        f.write(f"Momentum:  {best_result['momentum']}\n")
        f.write(f"Weight_decay: {best_result['weight_decay']}\n")
        f.write(f"box损失权重: {best_result['box']}\n")
        f.write(f"cls损失权重: {best_result['cls']}\n")
        f.write(f"dfl损失权重: {best_result['dfl']}\n")
        f.write(f"训练时间:  {best_result['train_time']:.1f}秒\n\n")

        # 写入所有结果排名
        f.write("📊 所有参数组合排名:\n")
        f.write("-" * 80 + "\n")
        f.write(
            f"{'排名':<4} {'参数名':<12} {'mAP50':<8} {'mAP50-95':<10} {'优化器':<8} {'学习率':<8} {'Momentum':<8}\n")
        f.write("-" * 80 + "\n")

        for i, result in enumerate(all_results, 1):
            f.write(f"{i:<4} {result['param_name']:<12} {result['map50']:.4f}   {result['map']:.4f}     "
                    f"{result['optimizer']:<8} {result['lr0']:<8} {result['momentum']:<8}\n")
        f.write("\n")

        # 写入详细参数分析
        f.write("🔍 详细参数分析:\n")
        f.write("-" * 80 + "\n")

        # 按优化器分析
        f.write("\n按优化器分析:\n")
        optimizers = set(r['optimizer'] for r in all_results)
        for opt in optimizers:
            opt_results = [r for r in all_results if r['optimizer'] == opt]
            best_opt = max(opt_results, key=lambda x: x['map50'])
            f.write(
                f"  {opt}: 最佳mAP50 = {best_opt['map50']:.4f} (学习率: {best_opt['lr0']}, Momentum: {best_opt['momentum']})\n")

        # 按学习率分析
        f.write("\n按学习率分析:\n")
        lr_groups = {}
        for result in all_results:
            lr = result['lr0']
            if lr not in lr_groups:
                lr_groups[lr] = []
            lr_groups[lr].append(result['map50'])

        for lr, maps in lr_groups.items():
            if maps:  # 确保列表不为空
                f.write(
                    f"  学习率 {lr}: 平均mAP50 = {sum(maps) / len(maps):.4f} (范围: {min(maps):.4f} - {max(maps):.4f})\n")

        # 按损失权重分析
        f.write("\n按边界框损失权重(box)分析:\n")
        box_groups = {}
        for result in all_results:
            box = result['box']
            if box not in box_groups:
                box_groups[box] = []
            box_groups[box].append(result['map50'])

        for box, maps in box_groups.items():
            if maps:
                f.write(
                    f"  box={box}: 平均mAP50 = {sum(maps) / len(maps):.4f} (范围: {min(maps):.4f} - {max(maps):.4f})\n")

        # 按分类损失权重分析
        f.write("\n按分类损失权重(cls)分析:\n")
        cls_groups = {}
        for result in all_results:
            cls_val = result['cls']
            if cls_val not in cls_groups:
                cls_groups[cls_val] = []
            cls_groups[cls_val].append(result['map50'])

        for cls_val, maps in cls_groups.items():
            if maps:
                f.write(
                    f"  cls={cls_val}: 平均mAP50 = {sum(maps) / len(maps):.4f} (范围: {min(maps):.4f} - {max(maps):.4f})\n")

        # 按DFL损失权重分析
        f.write("\n按DFL损失权重(dfl)分析:\n")
        dfl_groups = {}
        for result in all_results:
            dfl = result['dfl']
            if dfl not in dfl_groups:
                dfl_groups[dfl] = []
            dfl_groups[dfl].append(result['map50'])

        for dfl, maps in dfl_groups.items():
            if maps:
                f.write(
                    f"  dfl={dfl}: 平均mAP50 = {sum(maps) / len(maps):.4f} (范围: {min(maps):.4f} - {max(maps):.4f})\n")

        # 按Cosine LR分析
        f.write("\n按余弦学习率调度分析:\n")
        cos_lr_groups = {}
        for result in all_results:
            cos_lr = result['cos_lr']
            if cos_lr not in cos_lr_groups:
                cos_lr_groups[cos_lr] = []
            cos_lr_groups[cos_lr].append(result['map50'])

        for cos_lr, maps in cos_lr_groups.items():
            if maps:
                status = "启用" if cos_lr else "禁用"
                f.write(
                    f"  余弦学习率 {status}: 平均mAP50 = {sum(maps) / len(maps):.4f} (范围: {min(maps):.4f} - {max(maps):.4f})\n")
        # 写入建议
        f.write("\n💡 调优建议:\n")
        f.write("-" * 80 + "\n")
        f.write("1. 使用排名前3的参数组合进行最终训练\n")
        f.write("2. 如果时间允许，可以在最佳参数附近进行更精细的搜索\n")
        f.write("3. 考虑学习率对模型性能的重要影响\n")
        f.write("4. 注意损失权重(box, cls, dfl)的平衡配置\n")
        f.write("5. 不同的优化器可能需要不同的超参数设置\n")
        f.write("6. 余弦学习率调度可能有助于提高模型收敛性\n")
        f.write("7. 关注Momentum和Weight_decay对训练稳定性的影响\n")

    print(f"\n✅ 结果已保存到: {txt_path}")
    return txt_path


if __name__ == '__main__':
    print("开始超参数调优实验...")
    results = hyperparameter_tuning_single_fold()
    if results:
        print(f"\n🎉 调优完成！共完成 {len(results)} 个实验")
        best_result = max(results, key=lambda x: x['map50'])
        print(f"🏆 最佳mAP50: {best_result['map50']:.4f} (参数: {best_result['param_name']})")
    else:
        print("\n❌ 调优失败，没有有效结果")