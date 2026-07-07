# 模型验证/测试脚本（含ROC曲线、混淆矩阵、分类报告等）

from torchutils import *
from torchvision import datasets, models, transforms
import os.path as osp
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_curve, auc, RocCurveDisplay, cohen_kappa_score)
from sklearn.preprocessing import label_binarize
from itertools import cycle
from train import SELFMODEL

# 初始化设置
if torch.cuda.is_available():
    device = torch.device('cuda:1')
else:
    device = torch.device('cpu')
print(f'Using device: {device}')
# 固定随机种子
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False  # 保证可复现性

# 配置参数
data_path = r"D:\project_SJ\Resnet50d_classification\datasets_classify\cluster_four_origin_test_processed"
model_path = r"D:\project_SJ\Resnet50d_classification\runs\k_fold_processed\_fold_3\best_model_fold3.pth"
model_name = 'resnet50d'
img_size = 224
params = {
    'model': model_name,
    'img_size': img_size,
    'val_dir': data_path,
    'device': device,
    "name": "_resnet50_processed_",
    'batch_size': 64,
    'num_workers': 0,
    'num_classes': len(os.listdir(data_path)),
}


def plot_confusion_matrix(cm, class_names, normalize=True, title='Confusion Matrix', save_path=None):
    """绘制优化后的混淆矩阵"""
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(title)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


def plot_multiclass_roc(y_true, y_score, class_names, save_path):
    """绘制多类别ROC曲线及平均曲线"""
    y_true_bin = label_binarize(y_true, classes=np.arange(len(class_names)))
    fpr, tpr, roc_auc = {}, {}, {}    # 计算每个类别的ROC曲线
    for i in range(len(class_names)):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])   # 计算微平均ROC曲线
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(class_names))]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(class_names)):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= len(class_names)
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])   # 计算宏平均ROC曲线
    plt.figure(figsize=(10, 8))   # 绘制曲线
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red'])
    for i, color in zip(range(len(class_names)), colors):     # 绘制各类别曲线
        plt.plot(fpr[i], tpr[i], color=color, lw=1.5,
                 label='ROC {0} (AUC = {1:0.2f})'
                       "".format(class_names[i], roc_auc[i]))
    plt.plot(fpr["micro"], tpr["micro"],    # 绘制平均曲线
             label='Micro-average ROC (AUC = {0:0.2f})'
                   "".format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=3)
    plt.plot(fpr["macro"], tpr["macro"],
             label='Macro-average ROC (AUC = {0:0.2f})'
                   "".format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=3)
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multiclass ROC Curves')
    plt.legend(loc="lower right")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


def myval(val_loader, model, params, class_names):
    model.eval()
    stream = tqdm(val_loader)
    # 初始化收集变量
    test_real_labels = []
    test_pre_labels = []
    test_probs = []
    with torch.no_grad():
        for i, (images, target) in enumerate(stream, start=1):
            images = images.to(params['device'])
            target = target.to(params['device'])
            # 获取预测结果
            output = model(images)
            probs = torch.softmax(output, dim=1)
            preds = torch.argmax(probs, dim=1)
            # 收集数据
            test_real_labels.extend(target.cpu().numpy())
            test_pre_labels.extend(preds.cpu().numpy())
            test_probs.extend(probs.cpu().numpy())
    # 转换为numpy数组
    test_real_labels = np.array(test_real_labels)
    test_pre_labels = np.array(test_pre_labels)
    test_probs = np.array(test_probs)
    # 创建保存目录
    save_path = osp.join("record", params['name'])
    os.makedirs(save_path, exist_ok=True)
    # 1. 分类报告
    report = classification_report(
        test_real_labels, test_pre_labels,
        target_names=class_names, digits=4
    )
    with open(osp.join(save_path, "classification_report.txt"), "w") as f:
        f.write(report)
    # 2. 混淆矩阵
    cm = confusion_matrix(test_real_labels, test_pre_labels)
    plot_confusion_matrix(
        cm, class_names,
        title=f'Confusion Matrix ({model_name})',
        save_path=osp.join(save_path, f"confusion_matrix_{model_name}.png")
    )
    # 3. ROC曲线
    plot_multiclass_roc(
        test_real_labels, test_probs, class_names,
        save_path=osp.join(save_path, f"roc_curves_{model_name}.png")
    )
    return None



if __name__ == '__main__':
    # 数据加载
    data_transforms = get_torch_transforms(img_size=params["img_size"])
    val_dataset = datasets.ImageFolder(params["val_dir"], data_transforms['val'])
    class_names = val_dataset.classes
    print("Class names:", class_names)
    val_loader = DataLoader(
        val_dataset,
        batch_size=params['batch_size'],
        shuffle=False,  # 验证集不需要shuffle
        num_workers=params['num_workers'],
        pin_memory=True
    )
    # 模型加载
    model = SELFMODEL(
        model_name=params['model'],
        out_features=params['num_classes'],
        pretrained=False
    ).to(params['device'])
    model.load_state_dict(torch.load(model_path))
    # 执行验证
    myval(val_loader, model, params, class_names)




