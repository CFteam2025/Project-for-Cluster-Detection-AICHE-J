# 针对生成的数据集进行训练

import torch.optim
from torchutils import *
from torchvision import datasets, models, transforms
import os.path as osp
import os


if torch.cuda.is_available():
    device = torch.device('cuda:1')
else:
    device = torch.device('cpu')
# print(f'Using device: {device}')
# 固定随机种子，保证实验结果是可以复现的
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
data_path = r"D:\project_SJ\Resnet50d_classification\datasets_classify\cluster_four_origin.v10_split712"  # todo 数据集路径
# 注： 执行之前请先划分数据集
# 超参数设置
params = {
    'model': 'resnet50d',  # 选择预训练模型
    "img_size": 224,  # 图片输入大小
    "train_dir": osp.join(data_path, "train"),  # todo 训练集路径
    "val_dir": osp.join(data_path, "val"),  # todo 验证集路径
    'device': device,  # 设备
    "name": "_processed",
    'lr': 1e-3,  # 学习率
    'batch_size': 64,  # 批次大小
    'num_workers': 4,  # 进程
    'epochs': 200,  # 轮数
    "save_dir": "./runs/",  # todo 保存路径
    "pretrained": False,
     "num_classes": len(os.listdir(osp.join(data_path, "train"))),  # 类别数目, 自适应获取类别数目
    'weight_decay': 1e-5,  # 学习率衰减
    'max_grad_norm': 1.0,  # 梯度裁剪阈值
}


# 定义模型
class SELFMODEL(nn.Module):
    def __init__(self, model_name=params['model'], out_features=params['num_classes'],
                 pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)  # 从预训练的库中加载模型
        # self.model = timm.create_model(model_name, pretrained=pretrained, checkpoint_path="pretrained/resnet50d_ra2-464e36ba.pth")  # 从预训练的库中加载模型
        # classifier
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
        print(self.model)  # 返回模型

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
            output = model(images)    # 数据送入模型进行前向传播
            loss = criterion(output, target.long())   # 计算损失
        acc = accuracy(output, target)  # 计算准确率分数
        metric_monitor.update('Loss', loss.item())  # 更新损失
        metric_monitor.update('Accuracy', acc)  # 更新准确率
        optimizer.zero_grad()  # 清空学习率
        loss.backward()  # 损失反向传播
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), params['max_grad_norm'])
        optimizer.step()
        lr = adjust_learning_rate(optimizer, epoch, params, i, nBatch)  # 调整学习率
        stream.set_description(  # 更新进度条
            "Epoch: {epoch}. LR:{lr:.5e}. Train.      {metric_monitor}".format(
                epoch=epoch, lr=lr,
                metric_monitor=metric_monitor)
        )
    return metric_monitor.metrics['Accuracy']["avg"], metric_monitor.metrics['Loss']["avg"]  # 返回结果


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


# 展示训练过程的曲线
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
    save_path = osp.join(save_dir, "results.png")
    plt.savefig(save_path, dpi=100)


if __name__ == '__main__':
    data_transforms = get_torch_transforms(img_size=params["img_size"])  # 获取图像预处理方式
    train_transforms = data_transforms['train']  # 训练集数据处理方式
    valid_transforms = data_transforms['val']  # 验证集数据集处理方式
    train_dataset = datasets.ImageFolder(params["train_dir"], train_transforms)  # 加载训练集
    valid_dataset = datasets.ImageFolder(params["val_dir"], valid_transforms)  # 加载验证集
    train_loader = DataLoader(
        train_dataset, batch_size=params['batch_size'], shuffle=True,
        num_workers=params['num_workers'], pin_memory=True,
    )
    val_loader = DataLoader(
        valid_dataset, batch_size=params['batch_size'], shuffle=False,
        num_workers=params['num_workers'], pin_memory=True,
    )
    model = SELFMODEL(
        model_name=params['model'],
        out_features=params['num_classes'],
        pretrained=params['pretrained']
    ).to(params['device'])
    criterion = nn.CrossEntropyLoss().to(params['device'])
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=params['lr'],
        weight_decay=params['weight_decay'])
    accs = []
    losss = []
    val_accs = []
    val_losss = []
    best_acc = 0.0
    # 设置模型保存路径
    save_dir = osp.join(params['save_dir'], params['model'] + params['name'])
    if not osp.isdir(save_dir):
        os.makedirs(save_dir)
        print("save dir {} created".format(save_dir))
    # 创建CSV文件并写入表头
    csv_path = osp.join(save_dir, "training_log.csv")
    with open(csv_path, mode='w') as f:
        f.write("epoch,train_acc,train_loss,val_acc,val_loss\n")
    for epoch in range(1, params['epochs'] + 1):
        # 训练和验证
        acc, loss = train(train_loader, model, criterion, optimizer, epoch, params)
        val_acc, val_loss = validate(val_loader, model, criterion, epoch, params)
        # 记录指标
        accs.append(acc)
        losss.append(loss)
        val_accs.append(val_acc)
        val_losss.append(val_loss)
        # 写入CSV文件（追加模式）
        with open(csv_path, mode='a') as f:
            f.write(f"{epoch},{acc:.4f},{loss:.4f},{val_acc:.4f},{val_loss:.4f}\n")
        # 保存最佳模型和最新模型
        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(
                model.state_dict(),
                osp.join(save_dir, f"{params['model']}_best_weights.pth"))
        torch.save(
            model.state_dict(),
            osp.join(save_dir, f"latest_model.pth"))
    # 绘制损失和准确率曲线
    show_loss_acc(accs, losss, val_accs, val_losss, save_dir)
    print("类别列表:", train_dataset.classes)
    print("训练已完成，模型和训练日志保存在: {}".format(save_dir))
