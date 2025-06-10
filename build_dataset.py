import cv2
import requests
import base64
import json
from pathlib import Path
from dotenv import load_dotenv
import os
from ultralytics import YOLO
from question_classifier import QuestionClassifier, Question
import glob

load_dotenv()

class DatasetBuilder:
    """构建试卷数据集"""
    
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.app_id = os.getenv('MATHPIX_APP_ID')
        self.app_key = os.getenv('MATHPIX_API_KEY')
    
    def process_paper(self, paper_folder: str, paper_id: str) -> dict:
        """处理整份试卷"""
        paper_path = Path(paper_folder)
        if not paper_path.exists():
            return {'error': f'Paper folder not found: {paper_folder}'}
        
        print(f"📄 Processing paper: {paper_id}")
        
        # 获取所有页面图片，按页码数字排序
        page_files = sorted(paper_path.glob("page_*.png"), key=lambda p: int(p.stem.split('_')[1]))
        
        paper_result = {
            'paper_id': paper_id,
            'total_pages': len(page_files),
            'questions': [],
            'processing_log': []
        }
        
        classifier = QuestionClassifier(paper_id)
        all_analysis_results = []
        
        for page_file in page_files:
            print(f"  📑 Processing {page_file.name}...")
            
            page_results = self._process_page(page_file, page_file.stem)
            if page_results:
                all_analysis_results.extend(page_results)
                paper_result['processing_log'].append({
                    'page': page_file.name,
                    'detected_regions': len(page_results)
                })
        
        # 分类所有检测到的问题
        if all_analysis_results:
            questions = classifier.classify_regions(all_analysis_results)
            paper_result['questions'] = [self._question_to_dict(q) for q in questions]
        
        return paper_result
    
    def _process_page(self, image_path: Path, page_name: str) -> list:
        """处理单个页面"""
        try:
            # YOLO检测
            results = self.model(str(image_path))
            img = cv2.imread(str(image_path))
            
            analysis_results = []
            
            for r in results:
                if r.boxes is not None:
                    boxes = r.boxes.xyxy.cpu().numpy()
                    # 按y坐标排序（从上到下）
                    boxes = sorted(boxes, key=lambda x: x[1])
                    
                    for i, box in enumerate(boxes):
                        x1, y1, x2, y2 = map(int, box)
                        cropped_img = img[y1:y2, x1:x2]
                        
                        # Mathpix OCR
                        ocr_result = self._ocr_image(cropped_img)
                        
                        if ocr_result and 'text' in ocr_result:
                            analysis_results.append({
                                'text': ocr_result['text'],
                                'index': len(analysis_results),
                                'page': page_name,
                                'bbox': [x1, y1, x2, y2],
                                'ocr_confidence': ocr_result.get('confidence', 0)
                            })
            
            return analysis_results
            
        except Exception as e:
            print(f"    ❌ Error processing {image_path}: {e}")
            return []
    
    def _ocr_image(self, img_array):
        """对图像进行OCR"""
        try:
            # 编码图像
            _, buffer = cv2.imencode('.png', img_array)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            headers = {
                'app_id': self.app_id,
                'app_key': self.app_key,
                'Content-type': 'application/json'
            }
            
            data = {
                'src': f'data:image/png;base64,{image_base64}',
                'formats': ['text'],
            }
            
            response = requests.post('https://api.mathpix.com/v3/text', headers=headers, json=data)
            return response.json()
            
        except Exception as e:
            return {'error': str(e)}
    
    def _question_to_dict(self, question: Question) -> dict:
        """将Question对象转换为字典"""
        return {
            'main_question_id': question.main_question_id,
            'paper_info': question.paper_info,
            'full_id': question.get_full_id(),
            'main_part': {
                'text': question.main_part.text,
                'question_id': question.main_part.question_id,
                'type': question.main_part.part_type
            },
            'sub_parts': [
                {
                    'text': sub.text,
                    'question_id': sub.question_id,
                    'sub_letter': sub.sub_letter,
                    'full_id': question.get_full_id(sub.sub_letter),
                    'type': sub.part_type
                }
                for sub in question.sub_parts
            ],
            'total_parts': 1 + len(question.sub_parts)
        }

def main():
    """主函数"""
    print("🏗️  构建9709_s20_qp_41数据集")
    print("=" * 60)
    
    # 配置
    model_path = "models/pastpaper_detector_demo/weights/best.pt"
    paper_folder = "data/raw_images/9709_s20_qp_41"
    paper_id = "9709_s20_qp_41"
    output_file = "9709_s20_qp_41_dataset.json"
    
    # 构建数据集
    builder = DatasetBuilder(model_path)
    result = builder.process_paper(paper_folder, paper_id)
    
    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 数据集构建完成!")
    print(f"📁 输出文件: {output_file}")
    
    # 显示摘要
    if 'questions' in result:
        print(f"📊 统计信息:")
        print(f"   - 总页数: {result['total_pages']}")
        print(f"   - 检测到的主题: {len(result['questions'])}")
        
        total_sub_parts = sum(len(q['sub_parts']) for q in result['questions'])
        print(f"   - 总子题数: {total_sub_parts}")
        
        print(f"\n📋 题目列表:")
        for q in result['questions']:
            main_id = q['main_question_id']
            sub_count = len(q['sub_parts'])
            sub_letters = [s['sub_letter'] for s in q['sub_parts']]
            print(f"   Q{main_id}: {sub_count} 个子题 {sub_letters}")
    
    print(f"\n🔍 查看完整结果: cat {output_file}")

if __name__ == "__main__":
    main()