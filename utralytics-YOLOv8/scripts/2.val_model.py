from ultralytics import YOLO
import torch
import numpy as np


def main():
    # Load the model
    model_path = r'D:\project_SJ\CUP-cluster-detection\utralytics\scripts\runs\2.tuning_model\weights\best.pt'
    model = YOLO(model_path)
    # results = model.val(data=r"D:\project_SJ\ultralytics-main\datasets\dataset-v10-four-kfold\fold_4\dataset_fold_4.yaml", split="val",
    #                     conf=0.001, iou=0.7, name="val_final_augment_yolov8")
    results = model.val(data=r"D:\project_SJ\ultralytics-main\datasets\cluster_pic_4classes.yaml", split="test", conf=0.001, iou=0.7, name="test_base_yolov8")
    data = {
        "精度(Precision)": np.array(results.box.p).tolist(),
        "召回率(Recall)": np.array(results.box.r).tolist(),
        "F1分数": np.array(results.box.f1).tolist(),
        "mAP50": float(results.box.map50),
        "P曲线": np.array(results.box.p_curve).tolist(),
        "R曲线": np.array(results.box.r_curve).tolist(),
        "AP50-95": np.array(results.box.all_ap).tolist(),
        "类别索引": np.array(results.box.ap_class_index).tolist(),
        "每个类别的AP50": np.array(results.box.ap50).tolist(),  # 每个类别在IoU=0.5下的AP
    }
    with open('new_eval.txt', 'a', encoding='utf-8') as f:
        f.write(model_path)
        f.write("\n")
        for key, value in data.items():
            f.write(f"{key}: {value}\n")

    # Validate the model with a custom IOU threshold
    # iou_values = [0.5, 0.6, 0.7]  # Example IOU values
    # conf = 0.001
    # iou_values = [0.7]  # Example IOU values
    # for iou in iou_values:
    #     results = model.val(iou=iou, conf=conf)
    #     # print(f"IOU: {iou}, conf: {conf}")
        # # Print specific metrics
        # print("Average precision 50-95:", results.box.ap)
        # print("Average precision at IoU=0.50:", results.box.ap50)
        # print("F1 score:", results.box.f1)
        # print("Mean average precision 50-95:", results.box.map)
        # print("Mean average precision at IoU=0.50:", results.box.map50)
        # print("Mean average precision at IoU=0.75:", results.box.map75)
        # print("Mean precision:", results.box.mp)
        # print("Mean recall:", results.box.mr)
        # print("Precision:", results.box.p)
        # print("Recall:", results.box.r)



if __name__ == '__main__':
    # # 对于Windows多进程必要的支持
    # if torch.cuda.is_available() and torch.distributed.is_available():
    #     torch.distributed.init_process_group(backend='nccl')
    main()