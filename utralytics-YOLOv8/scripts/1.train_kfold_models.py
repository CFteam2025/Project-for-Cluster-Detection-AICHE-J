# 对k-fold的数据集自动训练

from ultralytics import YOLO
import os

def train_kfold_models():
    """训练5折交叉验证模型"""
    base_dataset_path = r"D:\project_SJ\ultralytics-main\datasets\dataset-v10-four-kfold -augment2"   # TODO
    for fold in range(1, 6):
        print(f"\\n开始训练第 {fold} 折...")
        # 当前折的配置路径
        config_path = os.path.join(base_dataset_path, f"fold_{fold}", f"dataset_fold_{fold}.yaml")
        # 初始化模型
        model = YOLO(r"./ultralytics/cfg/models/v8/yolov8m.yaml")
        # 训练模型
        model.train(
            data=config_path,
            epochs=200,
            device=[0,1],
            workers=0,
            batch=24,
            name=f"sj_yolov8_kfold_fold.augment2.v10_{fold}",
            pretrained=True,
            optimizer="SGD",
            cls=0.5,
            single_cls=False,
            patience=50,
            lr0=0.01,
            lrf=0.01,
            cache=True
        )
        print(f"第 {fold} 折训练完成!")

if __name__ == '__main__':
    train_kfold_models()
