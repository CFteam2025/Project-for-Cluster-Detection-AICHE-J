from torchutils import *
from torchvision import datasets, models, transforms
import os.path as osp
import os
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from torchvision import datasets


if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
print(f'Using device: {device}')
# 固定随机种子，保证实验结果是可以复现的
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
data_path = r"D:\project_SJ\Resnet50d_classification\datasets_classify\cluster_four_origin.v10" # todo 数据集路径
# 注： 执行之前请先划分数据集
# 超参数设置
params = {
    'model': 'resnet50d',  # 选择预训练模型
    "img_size": 224,  # 图片输入大小
    'device': device,  # 设备
    "name": "_origin_v10",
    'lr': 1e-4,  # 学习率
    'batch_size': 128,  # 批次大小
    'num_workers': 0,  # 进程
    'epochs': 200,  # 轮数
    "save_dir": "./runs/",  # todo 保存路径
    "pretrained": False,
     "num_classes": len(os.listdir(data_path)),  # 类别数目, 自适应获取类别数目
    'weight_decay': 1e-5,  # 学习率衰减
    'max_grad_norm': 1.0,  # 梯度裁剪阈值
}


# 定义模型
class SELFMODEL(nn.Module):
    def __init__(self, model_name=params['model'], out_features=params['num_classes'],
                 pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)  # 从预训练的库中加载模型
        if model_name[:3] == "res":
            n_features = self.model.fc.in_features  # 修改全连接层数目
            self.model.fc = nn.Linear(n_features, out_features)  # 修改为本任务对应的类别数目
        elif model_name[:3] == "vit":
            n_features = self.model.head.in_features  # 修改全连接层数目
            self.model.head = nn.Linear(n_features, out_features)  # 修改为本任务对应的类别数目
        else:
            n_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(n_features, out_features)
        # resnet修改最后的全链接层
        # print(self.model)  # 返回模型

    def forward(self, x):  # 前向传播
        x = self.model(x)
        return x


# 定义训练流程
def train(train_loader, model, criterion, optimizer, epoch, params):
    metric_monitor = MetricMonitor()  # 设置指标监视器
    model.train()  # 模型设置为训练模型
    nBatch = len(train_loader)
    stream = tqdm(train_loader)
    for i, (images, target) in enumerate(stream, start=1):  # 开始训练
        images = images.to(params['device'], non_blocking=True)  # 加载数据
        target = target.to(params['device'], non_blocking=True)  # 加载模型
        # 混合精度训练（如果可用）
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            output = model(images)
            loss = criterion(output, target.long())
        # output = model(images)  # 数据送入模型进行前向传播
        # loss = criterion(output, target.long())  # 计算损失
        acc = accuracy(output, target)  # 计算准确率分数
        metric_monitor.update('Loss', loss.item())  # 更新损失
        metric_monitor.update('Accuracy', acc)  # 更新准确率
        optimizer.zero_grad()  # 清空学习率
        loss.backward()  # 损失反向传播
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), params['max_grad_norm'])
        optimizer.step()  # 更新优化器
        lr = adjust_learning_rate(optimizer, epoch, params, i, nBatch)  # 调整学习率
        stream.set_description(  # 更新进度条
            "Epoch: {epoch}. LR:{lr:.5e}. Train.      {metric_monitor}".format(
                epoch=epoch, lr=lr,
                metric_monitor=metric_monitor)
        )
    return metric_monitor.metrics['Accuracy']["avg"], metric_monitor.metrics['Loss']["avg"]  # 返回结果

def show_loss_acc(acc, loss, val_acc, val_loss, save_dir):
    # 从history中提取模型训练集和验证集准确率信息和误差信息
    # 按照上下结构将图画输出
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    # 保存在savedir目录下。
    plt.savefig(save_dir, dpi=100)
    plt.close()


# 定义验证流程
def validate(val_loader, model, criterion, epoch, params):
    metric_monitor = MetricMonitor()  # 验证流程
    model.eval()  # 模型设置为验证格式
    stream = tqdm(val_loader)  # 设置进度条
    with torch.no_grad():  # 开始推理
        for i, (images, target) in enumerate(stream, start=1):
            images = images.to(params['device'], non_blocking=True)  # 读取图片
            target = target.to(params['device'], non_blocking=True)  # 读取标签
            output = model(images)  # 前向传播
            loss = criterion(output, target.long())  # 计算损失
            acc = accuracy(output, target)  # 计算acc
            metric_monitor.update('Loss', loss.item())  # 后面基本都是更新进度条的操作
            metric_monitor.update('Accuracy', acc)
            stream.set_description(
                "Epoch: {epoch}. Validation. {metric_monitor}".format(
                    epoch=epoch,
                    metric_monitor=metric_monitor)
            )
    return metric_monitor.metrics['Accuracy']["avg"], metric_monitor.metrics['Loss']["avg"]


def train_k_fold(params, data_transforms, num_folds=5):
    # 加载完整数据集（不划分训练/验证集，后续由 KFold 拆分）
    full_dataset = datasets.ImageFolder(data_path, data_transforms['train'])
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=params.get('seed', 42))
    fold_results = {
        'best_acc': [],
        'final_acc': [],
        'history': [],  # 每折的 accs, losss, val_accs, val_losss
        "fold_details": []
    }
    fold_dir = osp.join(params['save_dir'], f"k_fold"+params['name'])
    os.makedirs(fold_dir, exist_ok=True)
    # 初始化 CSV 文件（写入表头）
    csv_path = osp.join(fold_dir, "results.csv")
    with open(csv_path, 'w') as f:
        f.write("fold,epoch,train_acc,train_loss,val_acc,val_loss\n")
    for fold, (train_idx, val_idx) in enumerate(kfold.split(full_dataset)):
        print(f"\n===== Fold {fold + 1}/{num_folds} =====")
        # 划分数据集
        train_subset = Subset(full_dataset, train_idx)
        val_subset = Subset(full_dataset, val_idx)
        train_loader = DataLoader(
            train_subset, batch_size=params['batch_size'], shuffle=True,
            num_workers=params['num_workers'], pin_memory=True
        )
        val_loader = DataLoader(
            val_subset, batch_size=params['batch_size'], shuffle=False,
            num_workers=params['num_workers'], pin_memory=True
        )
        # 初始化模型（每折重新初始化）
        model = SELFMODEL(
            model_name=params['model'],
            out_features=params['num_classes'],
            pretrained=params['pretrained']
        ).to(params['device'])
        criterion = nn.CrossEntropyLoss().to(params['device'])
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=params['lr'],
            weight_decay=params['weight_decay']
        )
        # 存储当前折的训练日志
        accs, losss, val_accs, val_losss = [], [], [], []
        best_acc = 0.0
        # 创建当前折的保存目录
        fold_save_dir = osp.join(fold_dir, f"_fold_{fold}")
        os.makedirs(fold_save_dir, exist_ok=True)
        for epoch in range(1, params['epochs'] + 1):
            # 训练和验证
            acc, loss = train(train_loader, model, criterion, optimizer, epoch, params)
            val_acc, val_loss = validate(val_loader, model, criterion, epoch, params)
            # 记录日志
            accs.append(acc)
            losss.append(loss)
            val_accs.append(val_acc)
            val_losss.append(val_loss)
            # 保存到 CSV 文件（追加模式）
            with open(csv_path, 'a') as f:
                f.write(f"{fold},{epoch},{acc:.4f},{loss:.4f},{val_acc:.4f},{val_loss:.4f}\n")
            # 保存最佳模型和最新模型
            if val_acc >= best_acc:
                best_acc = val_acc
                torch.save(
                    model.state_dict(),
                    osp.join(fold_save_dir, f"best_model_fold{fold}.pth")
                )
            # 每个 epoch 保存一次最新模型（可选）
            torch.save(
                model.state_dict(),
                osp.join(fold_save_dir, f"latest_model_fold{fold}.pth")
            )
        pic_dir = osp.join(fold_save_dir, f'training_curve_fold{fold}.png')
        show_loss_acc(accs, losss, val_accs, val_losss, pic_dir)
        # 记录当前折的结果
        fold_results['best_acc'].append(best_acc)
        fold_results['final_acc'].append(val_accs[-1])
        fold_results['history'].append((accs, losss, val_accs, val_losss))
        fold_results['fold_details'].append({
            'fold': fold,
            'best_acc': best_acc,
            'final_acc': val_accs[-1],
            'train_acc': accs,
            'val_acc': val_accs
        })
    # 打印 K-Fold 汇总结果
    print("\n===== K-Fold 交叉验证结果 =====")
    print(f"每折最佳验证准确率: {fold_results['best_acc']}")
    print(f"平均最佳准确率: {sum(fold_results['best_acc']) / num_folds:.4f}")
    print(f"每折最终验证准确率: {fold_results['final_acc']}")
    print(f"平均最终准确率: {sum(fold_results['final_acc']) / num_folds:.4f}")
    # 将汇总结果保存为 CSV（可选）
    summary_df = pd.DataFrame({
        'fold': range(num_folds),
        'best_acc': fold_results['best_acc'],
        'final_acc': fold_results['final_acc']
    })
    summary_df.to_csv(osp.join(fold_save_dir, "summary.csv"), index=False)
    return fold_results

if __name__ == '__main__':
    # 获取数据预处理
    save_dir = osp.join(params['save_dir'], "k_fold")  # 设置模型保存路径
    data_transforms = get_torch_transforms(img_size=params["img_size"])
    # 运行 K-Fold 训练
    fold_results = train_k_fold(params, data_transforms, num_folds=5)
