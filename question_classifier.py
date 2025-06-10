import re
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class QuestionPart:
    """单个题目部分"""
    text: str
    question_id: str  # 如 "4", "4(a)", "4(b)"
    part_type: str   # "main" 或 "sub"
    sub_letter: Optional[str] = None  # 子题字母，如 "a", "b", "c"
    cropped_image_path: Optional[str] = None

@dataclass  
class Question:
    """完整题目，包含主题和所有子题"""
    main_question_id: str  # 如 "4"
    paper_info: str       # 如 "9709_s20_qp_11"
    main_part: QuestionPart
    sub_parts: List[QuestionPart]
    
    def get_full_id(self, sub_letter: str = None) -> str:
        """获取完整题目ID"""
        if sub_letter:
            return f"{self.paper_info}_Q{self.main_question_id}({sub_letter})"
        return f"{self.paper_info}_Q{self.main_question_id}"

class QuestionClassifier:
    """题目分类器，将检测到的区域组织成层次结构"""
    
    def __init__(self, paper_info: str):
        self.paper_info = paper_info  # 如 "9709_s20_qp_11"
    
    def classify_regions(self, analysis_results: List[dict]) -> List[Question]:
        """
        将分析结果分类组织
        analysis_results: [{'text': '...', 'index': 0}, ...]
        """
        questions = []
        current_main_question = None
        current_parent_letter = None  # 跟踪当前的父级子题字母 (a, b, c)
        
        for result in analysis_results:
            text = result['text']
            index = result['index']
            
            # 分析这个区域是主题还是子题
            part_info = self._analyze_question_part(text)
            
            if part_info['type'] == 'main':
                # 找到新的主题，保存之前的题目
                if current_main_question:
                    questions.append(current_main_question)
                
                # 重置状态
                current_parent_letter = None
                
                # 检查文本是否包含内嵌的(a)部分
                main_text, embedded_sub = self._split_main_and_first_sub(text)
                
                # 创建新的主题
                main_part = QuestionPart(
                    text=main_text,
                    question_id=part_info['main_id'],
                    part_type='main',
                    cropped_image_path=f"cropped_question_{index+1}.png"
                )
                
                current_main_question = Question(
                    main_question_id=part_info['main_id'],
                    paper_info=self.paper_info,
                    main_part=main_part,
                    sub_parts=[]
                )
                
                # 如果有内嵌的(a)部分，添加为第一个子题
                if embedded_sub:
                    sub_part = QuestionPart(
                        text=embedded_sub['text'],
                        question_id=f"{part_info['main_id']}({embedded_sub['letter']})",
                        part_type='sub',
                        sub_letter=embedded_sub['letter'],
                        cropped_image_path=f"cropped_question_{index+1}.png"
                    )
                    current_main_question.sub_parts.append(sub_part)
                    current_parent_letter = embedded_sub['letter']
                
            elif part_info['type'] == 'sub':
                # 子题，添加到当前主题
                if current_main_question:
                    sub_letter = part_info['sub_letter']
                    
                    # 处理嵌套子题的情况
                    if part_info.get('is_nested') and current_parent_letter:
                        # 这是一个 (ii) 类型的子题，需要组合为 (a)(ii)
                        sub_letter = f"{current_parent_letter}({sub_letter})"
                    elif len(sub_letter) == 1 and sub_letter in 'abcdef':
                        # 这是一个新的父级子题 (a), (b), (c)
                        current_parent_letter = sub_letter
                    elif '(' in sub_letter:
                        # 这是已经组合好的嵌套子题，如 a(i), b(i)
                        # 提取父级字母
                        parent_match = re.match(r'([a-z])\(', sub_letter)
                        if parent_match:
                            current_parent_letter = parent_match.group(1)
                    
                    sub_part = QuestionPart(
                        text=text,
                        question_id=f"{current_main_question.main_question_id}({sub_letter})",
                        part_type='sub',
                        sub_letter=sub_letter,
                        cropped_image_path=f"cropped_question_{index+1}.png"
                    )
                    current_main_question.sub_parts.append(sub_part)
                else:
                    print(f"⚠️  发现孤立的子题: {part_info['sub_letter']}")
        
        # 添加最后一个题目
        if current_main_question:
            questions.append(current_main_question)
        
        return questions
    
    def _split_main_and_first_sub(self, text: str) -> tuple:
        """
        将包含(a)部分的主题文本分割为主题部分和第一个子题
        返回: (main_text, embedded_sub_dict 或 None)
        """
        # 特殊格式：7 (a) Prove the identity... 或 7 (i) Prove the identity...
        # 检查是否以 "数字 (字母或罗马数字)" 开头
        special_pattern = r'^(\d+)\s+\(([a-z]+|i+)\)\s+(.*)'
        special_match = re.match(special_pattern, text.strip(), re.DOTALL)
        
        if special_match:
            question_num = special_match.group(1)
            sub_letter = special_match.group(2) 
            sub_content = special_match.group(3).strip()
            
            # 这种情况下，没有独立的主题描述，直接返回(a)作为第一个子题
            return f"{question_num}", {
                'letter': sub_letter,
                'text': f"({sub_letter}) {sub_content}"
            }
        
        # 普通格式：寻找换行后的(a)或(i)
        sub_pattern = r'\n\(([a-z]+|i+)\)\s+(.*?)(?=\n\([a-z]+|i+\)|\[|\Z)'
        match = re.search(sub_pattern, text, re.DOTALL)
        
        if match:
            # 找到(a)部分
            sub_letter = match.group(1)
            sub_text = match.group(2).strip()
            
            # 分割主题部分（(a)之前的内容）
            main_text = text[:match.start()].strip()
            
            return main_text, {
                'letter': sub_letter,
                'text': f"({sub_letter}) {sub_text}"
            }
        
        # 没有找到内嵌的子题
        return text, None
    
    def _analyze_question_part(self, text: str) -> dict:
        """分析文本确定是主题还是子题"""
        lines = text.strip().split('\n')
        
        # 检查第一行是否是纯数字（主题号）
        first_line = lines[0].strip()
        if re.match(r'^\d+$', first_line):
            return {
                'type': 'main',
                'main_id': first_line,
                'sub_letter': None
            }
        
        # 检查文本开头是否包含新的主题号（如 "9 The equation..." 或 "7 (a) Prove..." 或 "\( 7 \quad \) Let..."）
        main_number_match = re.match(r'^(\d+)\s+[A-Z(]', text.strip())
        if main_number_match:
            return {
                'type': 'main',
                'main_id': main_number_match.group(1),
                'sub_letter': None
            }
        
        # 检查LaTeX格式的主题号（如 "\( 7 \quad \) Let..."）
        latex_main_match = re.match(r'^\\\(\s*(\d+)\s+\\quad\s*\\\)\s+[A-Z]', text.strip())
        if latex_main_match:
            return {
                'type': 'main',
                'main_id': latex_main_match.group(1),
                'sub_letter': None
            }
        
        # 先清理LaTeX内容，避免 \( g(x) \) 这种干扰
        cleaned_text = self._remove_latex_content(text)
        
        # 首先检查是否包含 (a)...(i) 这种嵌套在同一文本块中的结构
        embedded_nested_match = re.search(r'^\s*\(([a-z])\)\s+.*?\n\(([i]+)\)\s', cleaned_text.strip(), re.DOTALL)
        if embedded_nested_match:
            main_letter = embedded_nested_match.group(1)
            sub_number = embedded_nested_match.group(2)
            return {
                'type': 'sub',
                'main_id': None,
                'sub_letter': f"{main_letter}({sub_number})"
            }
        
        # 检查嵌套结构：(a) (i) 或 (b) (ii) 等
        nested_match = re.search(r'^\s*\(([a-z])\)\s+\(([i]+)\)\s', cleaned_text.strip())
        if nested_match:
            main_letter = nested_match.group(1)
            sub_number = nested_match.group(2)
            return {
                'type': 'sub',
                'main_id': None,
                'sub_letter': f"{main_letter}({sub_number})"
            }
        
        # 检查是否以 (a), (b), (c) 等开头（一级子题）
        sub_match = re.search(r'^\s*\(([a-z])\)\s', cleaned_text.strip())
        if sub_match:
            return {
                'type': 'sub',
                'main_id': None,
                'sub_letter': sub_match.group(1)
            }
        
        # 检查是否以 (i), (ii), (iii) 等开头（二级子题，需要推断父级）
        roman_match = re.search(r'^\s*\((i+)\)\s', cleaned_text.strip())
        if roman_match:
            return {
                'type': 'sub',
                'main_id': None,
                'sub_letter': roman_match.group(1),
                'is_nested': True  # 标记为嵌套子题
            }
        
        # 检查文本中是否包含嵌套子题标记
        nested_pattern = r'(?<!\\\()\s*\(([a-z])\)\s+\(([i]+)\)\s+[A-Z]'
        nested_match = re.search(nested_pattern, cleaned_text)
        if nested_match:
            main_letter = nested_match.group(1)
            sub_number = nested_match.group(2)
            return {
                'type': 'sub', 
                'main_id': None,
                'sub_letter': f"{main_letter}({sub_number})"
            }
        
        # 检查文本中是否包含子题标记（更精确的模式）
        # 寻找独立的 (a), (b), (c)，不在LaTeX公式内
        sub_pattern = r'(?<!\\\()\s*\(([a-z])\)\s+[A-Z]'  # (a) 后面跟大写字母开头的句子
        sub_match = re.search(sub_pattern, cleaned_text)
        if sub_match:
            return {
                'type': 'sub', 
                'main_id': None,
                'sub_letter': sub_match.group(1)
            }
        
        # 默认判断为子题（如果没有明确标识）
        return {
            'type': 'sub',
            'main_id': None,
            'sub_letter': 'unknown'
        }
    
    def _remove_latex_content(self, text: str) -> str:
        """移除LaTeX内容，避免干扰题号识别"""
        # 移除 \( ... \) 格式的LaTeX
        text = re.sub(r'\\\([^)]+\\\)', '', text)
        # 移除 \[ ... \] 格式的LaTeX  
        text = re.sub(r'\\\[[^\]]+\\\]', '', text)
        # 移除 $ ... $ 格式的LaTeX
        text = re.sub(r'\$[^$]+\$', '', text)
        return text
    
    def build_search_index(self, questions: List[Question]) -> dict:
        """构建搜索索引，便于答案匹配"""
        search_index = {}
        
        for question in questions:
            # 主题索引
            main_id = question.get_full_id()
            search_index[main_id] = {
                'text': question.main_part.text,
                'type': 'main',
                'question': question
            }
            
            # 子题索引
            for sub_part in question.sub_parts:
                sub_id = question.get_full_id(sub_part.sub_letter)
                search_index[sub_id] = {
                    'text': sub_part.text,
                    'type': 'sub',
                    'question': question,
                    'sub_part': sub_part
                }
                
                # 也可以用简短的搜索词
                search_key = f"Q{question.main_question_id}({sub_part.sub_letter})"
                search_index[search_key] = search_index[sub_id]
        
        return search_index

def test_classifier():
    """测试分类器"""
    # 模拟分析结果
    mock_results = [
        {
            'text': '4\n\nThe diagram shows the graph of \\( y=\\mathrm{f}(x) \\), where \\( \\mathrm{f}(x)=\\frac{3}{2} \\cos 2 x+\\frac{1}{2} \\) for \\( 0 \\leqslant x \\leqslant \\pi \\).\n(a) State the range of f .\n[2]',
            'index': 0
        },
        {
            'text': 'A function g is such that \\( \\mathrm{g}(x)=\\mathrm{f}(x)+k \\), where \\( k \\) is a positive constant. The \\( x \\)-axis is a tangent to the curve \\( y=\\mathrm{g}(x) \\).\n(b) State the value of \\( k \\) and hence describe fully the transformation that maps the curve \\( y=\\mathrm{f}(x) \\) on to \\( y=\\mathrm{g}(x) \\).',
            'index': 1
        },
        {
            'text': '(c) State the equation of the curve which is the reflection of \\( y=\\mathrm{f}(x) \\) in the \\( x \\)-axis. Give your answer in the form \\( y=a \\cos 2 x+b \\), where \\( a \\) and \\( b \\) are constants.',
            'index': 2
        }
    ]
    
    print("🧪 测试题目分类器")
    print("=" * 60)
    
    classifier = QuestionClassifier("9709_s20_qp_11")
    questions = classifier.classify_regions(mock_results)
    
    print(f"📚 识别到 {len(questions)} 个完整题目:\n")
    
    for question in questions:
        print(f"🎯 主题 {question.main_question_id}:")
        print(f"   完整ID: {question.get_full_id()}")
        print(f"   主题内容: {question.main_part.text[:100]}...")
        
        print(f"   📋 子题 ({len(question.sub_parts)} 个):")
        for sub in question.sub_parts:
            print(f"      • {sub.question_id}: {sub.text[:80]}...")
        print()
    
    # 测试搜索索引
    print("🔍 搜索索引测试:")
    search_index = classifier.build_search_index(questions)
    
    # 模拟用户搜索
    test_queries = [
        "Q4(c)",
        "9709_s20_qp_11_Q4(b)",
        "State the equation of the curve"
    ]
    
    for query in test_queries:
        print(f"\n🔎 搜索: '{query}'")
        if query in search_index:
            result = search_index[query]
            print(f"   ✅ 找到: {result['type']} - {result['text'][:60]}...")
        else:
            # 模糊搜索
            matches = [k for k in search_index.keys() if query.lower() in search_index[k]['text'].lower()]
            if matches:
                print(f"   🎯 模糊匹配: {matches[:3]}")
            else:
                print("   ❌ 未找到")

if __name__ == "__main__":
    test_classifier()