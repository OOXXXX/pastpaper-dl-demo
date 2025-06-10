import os
import re
import requests
import json
import base64
import cv2
from pathlib import Path
from ultralytics import YOLO
from dotenv import load_dotenv

# --- 1. 加载环境变量和辅助函数 ---
load_dotenv()
MATHPIX_APP_ID = os.getenv("MATHPIX_APP_ID")
MATHPIX_API_KEY = os.getenv("MATHPIX_API_KEY")


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
        response.raise_for_status()
        return response.json().get("text", "OCR FAILED OR RETURNED EMPTY")
    except requests.exceptions.RequestException as e:
        return f"Error calling Mathpix API: {e}"


# --- 2. 核心处理器 ---
class DocumentProcessor:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.current_main_id = None
        self.last_sub_id = None

    def _extract_ids(self, text):
        """(健壮版) 从文本中提取主问题和子问题的ID"""
        main_ids, sub_ids = [], []
        p_main = re.compile(r"^\s*(\d+)\s*[.)]", re.MULTILINE)
        p_sub = re.compile(r"^\s*\(([a-z]+|[ivx]+)\)", re.IGNORECASE | re.MULTILINE)

        main_ids = p_main.findall(text)
        sub_ids = [match.lower() for match in p_sub.findall(text)]
        return main_ids, sub_ids

    def _increment_sub_id(self, sub_id):
        if sub_id and len(sub_id) == 1 and 'a' <= sub_id < 'z':
            return chr(ord(sub_id) + 1)
        return 'a'

    def process_document(self, sorted_page_paths, output_image_dir):
        all_questions_data = []
        doc_name = sorted_page_paths[0].parent.name

        for page_path in sorted_page_paths:
            print(f"--- Processing {page_path.name} ---")
            img = cv2.imread(str(page_path))
            if img is None:
                print(f"  > Warning: Could not read image {page_path.name}, skipping.")
                continue

            results = self.model(img)
            boxes = sorted(results[0].boxes, key=lambda box: box.xyxy[0][1])

            for i, box in enumerate(boxes):
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                cropped_img = img[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]

                b64_img = image_to_base64(cropped_img)
                ocr_text = call_mathpix_ocr(b64_img)

                main_ids, sub_ids = self._extract_ids(ocr_text)
                is_inferred = False

                if main_ids:
                    self.current_main_id = main_ids[0]
                    self.last_sub_id = None

                if sub_ids:
                    self.last_sub_id = sub_ids[-1]
                elif not main_ids:
                    is_inferred = True
                    self.last_sub_id = self._increment_sub_id(self.last_sub_id)

                final_id = self.current_main_id or ""
                if self.last_sub_id:
                    final_id += f"({self.last_sub_id})"

                if not final_id:
                    print(f"  > Warning: Could not determine ID for a box on {page_path.name}. Skipping.")
                    continue

                image_filename = f"{doc_name}_{page_path.stem}_q{final_id}.png"
                image_save_path = output_image_dir / image_filename
                cv2.imwrite(str(image_save_path), cropped_img)

                question_data = {
                    "unique_id": f"{doc_name}_{final_id}",
                    "source_paper": doc_name,
                    "question_id": final_id,
                    "text": ocr_text,
                    "image_path": str(image_save_path.relative_to(PROJECT_ROOT)),
                    "is_inferred": is_inferred
                }
                all_questions_data.append(question_data)
                print(f"  > Detected question. Assigned ID: {final_id}{' (inferred)' if is_inferred else ''}")

        return all_questions_data


# --- 3. 脚本执行入口 ---
if __name__ == '__main__':
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent

    model_path = PROJECT_ROOT / "models" / "pastpaper_detector_demo" / "weights" / "best.pt"
    output_image_dir = PROJECT_ROOT / "data" / "processed_questions" / "images"
    output_image_dir.mkdir(parents=True, exist_ok=True)

    # --- 重点：我们只测试 9709_s20_qp_11 这一份试卷 ---
    paper_to_test = "9709_s20_qp_11"
    image_dir = PROJECT_ROOT / "data" / "raw_images" / paper_to_test

    if not image_dir.is_dir():
        print(f"❌ 测试失败：请确保文件夹 {image_dir} 存在，并且其中包含页面图片。")
    else:
        print(f"\n{'=' * 20} Processing Document: {image_dir.name} {'=' * 20}")

        sorted_pages = sorted(image_dir.glob("*.png"), key=lambda p: int(p.stem.split('_')[-1]))

        processor = DocumentProcessor(model_path=model_path)
        document_questions = processor.process_document(sorted_pages, output_image_dir)

        # 将本次测试的结果保存到一个独立的JSON文件中，方便检查
        output_json_path = PROJECT_ROOT / "data" / "processed_questions" / f"{paper_to_test}_dataset.json"
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(document_questions, f, indent=2, ensure_ascii=False)

        print(f"\n✅ Test complete. Successfully created structured dataset at {output_json_path}")