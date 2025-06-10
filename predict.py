# predict.py
from ultralytics import YOLO
import cv2
from pathlib import Path

# --- 1. 加载您训练好的模型 ---
# 路径和您日志中显示的一致
MODEL_PATH = 'models/pastpaper_detector_demo/weights/best.pt'
model = YOLO(MODEL_PATH)

# --- 2. 指定您要测试的图片路径 ---
# 把它换成您准备好的那张新图片的实际路径
test_image_path = "data/raw_images/9709_s20_qp_11/page_5.png"

# --- 3. 进行推理并可视化结果 ---
if not Path(test_image_path).exists():
    print(f"Error: Test image not found at {test_image_path}")
else:
    # 模型进行预测
    results = model(test_image_path)

    # 加载原始图片以便画框
    img = cv2.imread(test_image_path)

    # 遍历所有检测到的框
    for r in results:
        for box in r.boxes:
            # 获取坐标
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            # 在图片上画出绿色的矩形框
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 将画好框的图片保存下来
    output_path = "croptest/prediction_result.png"
    cv2.imwrite(output_path, img)
    print(f"✅ Prediction result saved to {output_path}. Go and check it out!")