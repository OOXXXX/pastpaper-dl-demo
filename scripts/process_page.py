import os
import re
import requests
import json
import base64
import cv2
from pathlib import Path
from ultralytics import YOLO
from dotenv import load_dotenv

# --- 1. 加载环境变量 (安全的做法) ---
load_dotenv()
MATHPIX_APP_ID = os.getenv("MATHPIX_APP_ID")
MATHPIX_API_KEY = os.getenv("MATHPIX_API_KEY")


# --- 2. 定义辅助函数 ---

def image_to_base64(image):
    """将OpenCV图片对象转换为Base64字符串"""
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer).decode('utf-8')


def call_mathpix_ocr(image_b64):
    """调用Mathpix API进行OCR识别"""
    if not (MATHPIX_APP_ID and MATHPIX_API_KEY):
        raise ValueError("请确保.env文件中设置了MATHPIX_APP_ID和MATHPIX_API_KEY")

    url = "https://api.mathpix.com/v3/text"
    headers = {
        "app_id": MATHPIX_APP_ID,
        "app_key": MATHPIX_API_KEY,
        "Content-type": "application/json"
    }
    payload = {
        "src": f"data:image/png;base64,{image_b64}",
        "formats": ["text"]
    }
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()  # 如果请求失败则抛出异常
        # 我们只关心识别出的纯文本
        return response.json().get("text", "")
    except requests.exceptions.RequestException as e:
        print(f"Error calling Mathpix API: {e}")
        return ""


# 替换旧的 extract_question_id 函数
# 替换旧的 extract_question_id 函数
def extract_question_ids(text):
    """
    使用更健壮的正则表达式，从一个文本块中提取所有题号。
    返回一个ID列表，例如 ['4', 'a']。
    """
    # 模式1: 匹配开头的 "4", "10." 等数字题号。
    # 要求后面必须跟一个点或至少一个空格，以避免匹配分数中的数字。
    # re.MULTILINE让^可以匹配每一行的开头。
    p_main = re.compile(r"^\s*(\d+)[.\s]", re.MULTILINE)

    # 模式2: 匹配 "(a)", "(b)", "c)" 等子问题题号。
    p_sub = re.compile(r"\(?([a-z]{1,2})\)?[.\s]", re.IGNORECASE)

    main_ids = [match.strip() for match in p_main.findall(text)]
    sub_ids = [match.strip().lower() for match in p_sub.findall(text)]

    # 合并并返回所有找到的唯一ID
    return sorted(list(set(main_ids + sub_ids)))


# 替换旧的 increment_question_id 函数
def increment_question_id(q_id):
    """更智能地递增题号"""
    if not q_id:
        return None

        # 如果上一个是数字 (如 '4')，下一个应该是 'a'
    if q_id.isdigit():
        return 'a'

    # 如果上一个是单个字母 (如 'a')，下一个是 'b'
    if len(q_id) == 1 and 'a' <= q_id < 'z':
        return chr(ord(q_id) + 1)

    # 其他复杂情况（如 'iv' -> 'v'）暂不处理
    return None

# --- 3. 主处理流程 ---

# 替换旧的 process_page_with_context 函数
def process_page_with_context(image_path, yolo_model, last_id_from_prev_page=None):
    """
    处理单张页面图片，并利用上下文推断题号 (最终优化版)。
    """
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return []

    results = yolo_model(img)
    boxes = sorted(results[0].boxes, key=lambda box: box.xyxy[0][1])

    # --- 数据首次处理 ---
    all_boxes_info = []
    for box in boxes:
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        cropped_img = img[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
        b64_img = image_to_base64(cropped_img)
        ocr_text = call_mathpix_ocr(b64_img)

        # 使用新的函数，获取ID列表
        ids = extract_question_ids(ocr_text)

        all_boxes_info.append({
            "ids": ids,
            "text": ocr_text.strip(),
        })

    # --- 上下文推断与ID最终确定 ---
    final_results = []
    current_main_id = None
    last_sub_id = None

    # 如果上一页有题号，初始化
    if last_id_from_prev_page:
        if last_id_from_prev_page.isdigit():
            current_main_id = last_id_from_prev_page
        else:
            last_sub_id = last_id_from_prev_page

    for i, box_info in enumerate(all_boxes_info):
        final_id = None
        is_inferred = False

        if box_info["ids"]:
            # 如果框内识别出了ID
            for q_id in box_info["ids"]:
                if q_id.isdigit():
                    current_main_id = q_id
                    last_sub_id = None  # 主题号出现，重置子题号
                else:
                    last_sub_id = q_id
            # 我们将框的主要ID设为它包含的最后一个ID
            final_id = f"{current_main_id}{last_sub_id}" if last_sub_id else current_main_id
        else:
            # 如果框内没有识别出ID，进行推断
            if last_sub_id:  # 优先根据上一个子题号推断
                final_id_sub = increment_question_id(last_sub_id)
                if final_id_sub:
                    last_sub_id = final_id_sub
                    final_id = f"{current_main_id}{last_sub_id}"
                    is_inferred = True
            elif current_main_id:  # 如果没有子题号，根据主推断
                last_sub_id = "a"
                final_id = f"{current_main_id}{last_sub_id}"
                is_inferred = True

        final_results.append({
            "id": final_id,
            "text": box_info["text"],
            "inferred": is_inferred
        })

    return final_results


if __name__ == '__main__':
    # --- 4. 使用示例 ---

    # 加载您训练好的YOLO模型
    yolo_model = YOLO('../models/pastpaper_detector_demo/weights/best.pt')

    # 指定要测试的图片
    test_image_path = Path("../data/raw_images/9709_s20_qp_11/page_5.png")

    # 模拟一个从上一页传来的最后一个题号 (用于测试跨页逻辑)
    # 假设 page_4 的最后一个题号是 "3(b)"
    prev_page_last_id = "3(b)"

    print(f"--- Processing page: {test_image_path.name} ---")
    final_results = process_page_with_context(test_image_path, yolo_model, last_id_from_prev_page=prev_page_last_id)

    # 打印最终结果
    print("\n--- Final Identified Questions ---")
    for box in final_results:
        status = " (inferred)" if box['inferred'] else ""
        print(f"ID: {box['id']}{status}\nText: {box['text'][:50]}...\n")