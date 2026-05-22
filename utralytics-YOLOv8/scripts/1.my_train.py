from ultralytics import YOLO
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def train():
    model = YOLO(r"D:\project_SJ\ultralytics-main\ultralytics\cfg\models\v8\yolov8m.yaml")
    # model.load(",,,.pt")  # load a pretrained model
    print("模型配置：", model.model.yaml)
    print("训练参数：", model.overrides)  # 打印训练参数
    model.train(data=r"D:\project_SJ\ultralytics-main\datasets\dataset-v10-four-kfold\fold_4\dataset_fold_4.yaml",    # TODO
                epochs=70,
                device=[0, 1],
                workers=4,
                optimizer="SGD",
                batch=24,
                project="runs",
                name="augment_model_100",
                pretrained=True,
                single_cls=False,
                weight_decay=0.0005,
                momentum=0.937,
                lr0=0.01,  # 学习率优化
                lrf=0.01,
                box=6.5,
                cls=0.75,
                dfl=1,
                cache=False,
                close_mosaic=20,
                patience=50
                )

if __name__ == '__main__':
    train()
