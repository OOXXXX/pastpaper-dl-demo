from ultralytics import YOLO

def main():
    # 加载预训练的YOLOv8n模型
    model = YOLO('yolov8n.pt')

    # 开始训练
    # 注意: data路径应指向您解压数据集后得到的 `data.yaml` 文件
    model.train(
        data='data/annotated_dataset/data.yaml',
        epochs=80,  # 对于小数据集，80轮通常足够
        imgsz=640,
        project='models', # 将训练结果保存在 models/ 文件夹下
        name='pastpaper_detector_demo'
    )

if __name__ == '__main__':
    main()