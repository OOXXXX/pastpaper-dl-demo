from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from ultralytics import YOLO
import cv2
import numpy as np
import io
import base64
import os

app = FastAPI(title="Past Paper Question Detector API")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# 加载您训练好的模型
MODEL_PATH = "../models/pastpaper_detector_demo/weights/best.pt"
model = YOLO(MODEL_PATH)

@app.post("/detect/")
async def detect_and_crop(file: UploadFile = File(...)):
    """
    接收一张图片，检测其中的题目，并返回裁剪后的题目图片列表 (Base64编码)
    """
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 模型推理
    results = model(img)

    cropped_images_base64 = []
    for r in results:
        for box in r.boxes:
            # 获取坐标并裁剪
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            cropped_img = img[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
            
            # 将图片编码为Base64字符串
            _, buffer = cv2.imencode('.png', cropped_img)
            base64_str = base64.b64encode(buffer).decode('utf-8')
            cropped_images_base64.append(base64_str)

    return {"detected_questions": cropped_images_base64}

@app.get("/")
async def root():
    return FileResponse('static/index.html')

@app.get("/health")
def health_check():
    return {"message": "Welcome to the Question Detector API!", "status": "healthy"}