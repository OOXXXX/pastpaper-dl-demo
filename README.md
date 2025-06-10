# 🎓 Past Paper Question Detector & Visualizer

A complete AI-powered system for automatically detecting, extracting, and visualizing questions from Cambridge A-Level Mathematics past papers using YOLOv8.

## 📋 Project Overview

This project implements an end-to-end pipeline that:
1. **Converts PDF past papers to images** with intelligent filtering
2. **Trains a YOLOv8 model** to detect individual questions
3. **Extracts question text using OCR** and organizes into structured datasets
4. **Classifies questions hierarchically** (main questions and sub-parts)
5. **Provides a web-based visualizer** with LaTeX math rendering
6. **Offers a FastAPI service** for real-time question detection and cropping

## 🚀 Features

- ✅ **Smart PDF Processing**: Automatically converts PDFs to images while filtering out cover pages and blank pages
- ✅ **Question Detection**: YOLOv8-based model trained on Cambridge A-Level Mathematics papers
- ✅ **OCR Integration**: Extracts text from detected question regions using EasyOCR
- ✅ **Intelligent Question Classification**: Hierarchical parsing of main questions and sub-parts (a, b, c) with support for nested structures like (a)(i), (a)(ii)
- ✅ **Structured Dataset Generation**: Creates JSON datasets with organized question content
- ✅ **Academic Web Visualizer**: Clean, academic-style HTML viewer with MathJax LaTeX rendering
- ✅ **Interactive Web Interface**: Modern, Claude-inspired UI for question detection and download
- ✅ **Multi-Paper Support**: Dynamic selection between different past papers
- ✅ **REST API**: FastAPI service for uploading images and getting detected questions
- ✅ **Real-world Dataset**: Trained on actual CAIE Mathematics (9709) 2020 Summer papers

## 📁 Project Structure

```
pastpaper-dl-demo/
├── 📂 data/
│   ├── 📂 raw_images/           # Converted PDF pages (16 different papers)
│   │   ├── 9709_s20_qp_11/      # Paper 11 pages
│   │   ├── 9709_s20_qp_12/      # Paper 12 pages
│   │   └── ...                  # More papers (11-63)
│   ├── 📂 annotated_dataset/    # Roboflow exported training data
│   │   ├── data.yaml            # Dataset configuration
│   │   ├── train/ (27 images)   # Training set
│   │   ├── valid/ (8 images)    # Validation set
│   │   └── test/ (3 images)     # Test set
│   └── 📂 processed_questions/  # OCR-extracted question crops
├── 📂 scripts/
│   ├── prepare_data.py          # PDF to image conversion script
│   ├── process_page.py          # Individual page processing
│   └── debug_ocr.py             # OCR debugging utilities
├── 📂 api/
│   ├── main.py                  # FastAPI service for question detection
│   └── static/
│       └── index.html           # Modern web interface for question detection
├── 📂 models/
│   └── pastpaper_detector_demo/ # Trained YOLOv8 model and metrics
│       └── weights/
│           ├── best.pt          # Best performing model
│           └── last.pt          # Latest checkpoint
├── 📂 croptest/                 # Question cropping examples
│   ├── crop_and_analyze.py      # Cropping and analysis script
│   └── cropped_question_*.png   # Sample cropped questions
├── build_dataset.py             # Main dataset generation script
├── question_classifier.py       # Question hierarchical classification
├── train.py                     # Model training script
├── predict.py                   # Test script for single image prediction
├── academic_viewer.html         # Academic-style question visualizer
├── 9709_s20_qp_*_dataset.json   # Generated structured datasets (11, 21, 31, 41)
└── 📊 Training Results & Visualizations
```

## 🛠️ Setup & Installation

### 1. Clone & Setup Environment
```bash
cd pastpaper-dl-demo
source venv/bin/activate  # Virtual environment already configured
```

### 2. Dependencies
All required packages are pre-installed:
- `ultralytics` (YOLOv8)
- `pymupdf` (PDF processing)
- `opencv-python` (Image processing)
- `easyocr` (Text extraction)
- `fastapi` + `uvicorn` (API service)

## 🎯 Usage Guide

### 📄 Convert PDFs to Images
```bash
cd scripts
python prepare_data.py
```
**Features:**
- Automatically skips cover pages (page 1)
- Filters out blank pages and instruction pages
- Creates organized folders for each paper
- High-quality 300 DPI conversion

### 🤖 Train the Model
```bash
python train.py
```
**Training Details:**
- Model: YOLOv8n (nano - fast inference)
- Epochs: 80
- Dataset: 38 annotated images with question bounding boxes
- Class: Single class "question"

### 📊 Generate Question Datasets
```bash
python build_dataset.py
```
**Features:**
- Detects questions using trained YOLOv8 model
- Extracts text using EasyOCR
- Classifies questions into hierarchical structure
- Supports nested sub-questions like (a)(i), (a)(ii), (b)(i), (b)(ii)
- Generates structured JSON datasets
- Creates individual cropped question images

### 🌐 View Questions in Academic Style
Open `academic_viewer.html` in your browser for:
- Clean, academic typography (Times New Roman)
- LaTeX math rendering with MathJax
- Dynamic paper selection (qp_11, qp_21, qp_31, qp_41)
- Organized question hierarchy
- No redundant information display

### 🔍 Test Single Image
```bash
python predict.py
```
Generates `prediction_result.png` with detected questions highlighted.

### 🌐 Run API Service & Web Interface
```bash
cd api
uvicorn main:app --reload
```
Then visit: `http://127.0.0.1:8000` for the web interface or `http://127.0.0.1:8000/docs` for API documentation.

**Web Interface Features:**
- Clean, modern UI inspired by Claude's design language
- Drag-and-drop image upload
- Real-time question detection and visualization
- Individual question download functionality
- Responsive design for desktop and mobile

**API Endpoints:**
- `POST /detect/` - Upload image, get cropped questions as Base64
- `GET /health` - Health check
- `GET /` - Serves the web interface

## 📊 Model Performance

The trained model successfully detects questions from Cambridge A-Level Mathematics papers with:
- **Training Set**: 27 images with comprehensive question annotations
- **Validation Set**: 8 images for model validation
- **Test Set**: 3 images for final evaluation

Training artifacts include precision/recall curves, confusion matrices, and sample predictions.

## 🎓 Dataset Details

**Source**: Cambridge Assessment International Education (CAIE)
- **Subject**: A-Level Mathematics (9709)
- **Session**: 2020 Summer
- **Papers**: 16 different question papers (variants 11-63)
- **Total Pages**: ~160 pages of mathematical content
- **Annotation**: Roboflow platform with question bounding boxes

## 📈 Generated Datasets

The system has successfully processed and generated structured datasets for:

- **9709_s20_qp_11_dataset.json**: Pure Mathematics 1 (11 questions, 20 sub-parts)
- **9709_s20_qp_21_dataset.json**: Pure Mathematics 2 
- **9709_s20_qp_31_dataset.json**: Pure Mathematics 3 (10 questions with complex nested structures)
- **9709_s20_qp_41_dataset.json**: Mechanics 1

Each dataset includes:
- Hierarchical question structure
- LaTeX-formatted mathematical expressions
- Individual question cropping coordinates
- Complete sub-question classification

## 🔮 Future Enhancements

1. **Answer Matching**: Parse mark schemes and link questions to answers
2. **Enhanced OCR**: Improve mathematical expression recognition using Mathpix
3. **Database Integration**: Store questions and answers in structured format
4. **Multi-subject Support**: Extend to other A-Level subjects (Physics, Chemistry)
5. **Question Classification**: Categorize by topic/difficulty using NLP
6. **Export Formats**: Add support for Word, PDF, and Anki flashcard exports

## 🤝 Contributing

This is a demonstration project showing practical AI application in education technology. Feel free to:
- Fork and experiment with different models
- Test on other examination boards
- Improve the preprocessing pipeline
- Add new features like question classification

## 📄 License

This project is for educational demonstration purposes. Past paper content belongs to Cambridge Assessment International Education.

---

**Built with ❤️ using YOLOv8, FastAPI, and modern AI techniques**