import fitz  # PyMuPDF
from pathlib import Path

# --- 配置您的本地路径 ---
# 输入目录：包含所有PDF试卷的文件夹
QP_PDF_DIR = Path("/Users/patrick/Desktop/Container/CAIE/Alevel/Mathematics (9709)/2020/Summer/Question_Paper/")
# 输出根目录：所有图片将存放在这里
OUTPUT_ROOT_DIR = Path("../data/raw_images/")
# 要过滤的关键词
IGNORE_KEYWORDS = ["BLANK PAGE", "Additional Page", "INSTRUCTIONS"]


def process_all_pdfs_to_folders(pdf_dir, output_root_dir):
    """
    遍历指定目录下的所有PDF文件，
    为每个PDF创建一个独立的子文件夹，并将其页面转换为图片存入其中。
    """
    # 确保根输出目录存在
    output_root_dir.mkdir(parents=True, exist_ok=True)

    # 检查输入目录是否存在
    if not pdf_dir.exists():
        print(f"Error: Input directory not found at {pdf_dir}")
        return

    # 遍历所有PDF文件
    for pdf_path in pdf_dir.glob("*.pdf"):
        print(f"Processing {pdf_path.name}...")

        # 1. 为当前PDF创建一个子文件夹
        # 使用pdf_path.stem获取不带.pdf后缀的文件名作为文件夹名
        pdf_output_folder = output_root_dir / pdf_path.stem
        pdf_output_folder.mkdir(exist_ok=True)

        doc = fitz.open(pdf_path)

        for i, page in enumerate(doc):
            # 规则1: 固定跳过第一页
            if i == 0:
                print(f"  - Skipping page {i + 1} (Cover Page)")
                continue

            # 规则2: 检查页面文本，过滤无用页面
            text = page.get_text("text")
            if any(keyword in text for keyword in IGNORE_KEYWORDS):
                print(f"  - Skipping page {i + 1} (Contains ignored keyword)")
                continue

            # 2. 将图片保存在新建的子文件夹中
            pix = page.get_pixmap(dpi=300)
            output_path = pdf_output_folder / f"page_{i + 1}.png"
            pix.save(output_path)

        doc.close()


if __name__ == "__main__":
    process_all_pdfs_to_folders(QP_PDF_DIR, OUTPUT_ROOT_DIR)
    print("\n✅ All PDFs have been converted and organized into individual folders.")