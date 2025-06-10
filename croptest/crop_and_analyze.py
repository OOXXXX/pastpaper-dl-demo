import cv2
import requests
import base64
from dotenv import load_dotenv
import os
from ultralytics import YOLO

load_dotenv()

def crop_detected_questions(image_path, model_path):
    """使用YOLO模型检测问题区域并裁剪"""
    model = YOLO(model_path)
    results = model(image_path)
    
    img = cv2.imread(image_path)
    cropped_regions = []
    
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()  # 获取检测框坐标
        # 按y坐标排序（从上到下）
        boxes = sorted(boxes, key=lambda x: x[1])
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            cropped_img = img[y1:y2, x1:x2]
            cropped_regions.append({
                'image': cropped_img,
                'box': (x1, y1, x2, y2),
                'index': i
            })
    
    return cropped_regions

def analyze_cropped_region(cropped_img):
    """用Mathpix分析单个裁剪区域"""
    app_id = os.getenv('MATHPIX_APP_ID')
    app_key = os.getenv('MATHPIX_API_KEY')
    
    # 将opencv图像编码为base64
    _, buffer = cv2.imencode('.png', cropped_img)
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    
    headers = {
        'app_id': app_id,
        'app_key': app_key,
        'Content-type': 'application/json'
    }
    
    data = {
        'src': f'data:image/png;base64,{image_base64}',
        'formats': ['text'],
    }
    
    try:
        response = requests.post('https://api.mathpix.com/v3/text', headers=headers, json=data)
        return response.json()
    except Exception as e:
        return {'error': str(e)}

if __name__ == "__main__":
    # 使用原始图片和模型，重新检测并裁剪
    original_image = "data/raw_images/9709_s20_qp_11/page_5.png"
    model_path = "../models/pastpaper_detector_demo/weights/best.pt"
    
    print("🔍 检测并裁剪问题区域...")
    cropped_regions = crop_detected_questions(original_image, model_path)
    
    print(f"✅ 检测到 {len(cropped_regions)} 个问题区域")
    
    for i, region in enumerate(cropped_regions):
        print(f"\n📝 分析区域 {i+1}:")
        print("-" * 30)
        
        # 保存裁剪图片用于调试
        cv2.imwrite(f"cropped_question_{i+1}.png", region['image'])
        
        # 用Mathpix分析
        result = analyze_cropped_region(region['image'])
        
        if 'error' in result:
            print(f"❌ 错误: {result['error']}")
        else:
            text = result.get('text', '').strip()
            print(f"识别文本:\n{text}")
            
            # 简单提取题号
            lines = text.split('\n')
            first_line = lines[0].strip() if lines else ""
            print(f"首行: '{first_line}'")