# 🎓 Past Paper Question Detector

A complete AI-powered system for automatically detecting and cropping questions from Cambridge A-Level Mathematics past papers using YOLOv8.

## 📋 Project Overview

This project implements an end-to-end pipeline that:
1. **Converts PDF past papers to images** with intelligent filtering
2. **Trains a YOLOv8 model** to detect individual questions
3. **Provides a FastAPI service** for real-time question detection and cropping
4. **Demonstrates practical AI application** for educational content processing

## 🚀 Features

- ✅ **Smart PDF Processing**: Automatically converts PDFs to images while filtering out cover pages and blank pages
- ✅ **Question Detection**: YOLOv8-based model trained on Cambridge A-Level Mathematics papers
- ✅ **Organized Output**: Each paper gets its own folder with properly numbered pages
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
│   └── 📂 annotated_dataset/    # Roboflow exported training data
│       ├── data.yaml            # Dataset configuration
│       ├── train/ (27 images)   # Training set
│       ├── valid/ (8 images)    # Validation set
│       └── test/ (3 images)     # Test set
├── 📂 scripts/
│   └── prepare_data.py          # PDF to image conversion script
├── 📂 api/
│   └── main.py                  # FastAPI service for question detection
├── 📂 models/
│   └── pastpaper_detector_demo/ # Trained YOLOv8 model and metrics
│       └── weights/
│           ├── best.pt          # Best performing model
│           └── last.pt          # Latest checkpoint
├── train.py                     # Model training script
├── predict.py                   # Test script for single image prediction
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

### 🔍 Test Single Image
```bash
python predict.py
```
Generates `prediction_result.png` with detected questions highlighted.

### 🌐 Run API Service
```bash
cd api
uvicorn main:app --reload
```
Then visit: `http://127.0.0.1:8000/docs` for interactive API documentation.

**API Endpoints:**
- `POST /detect/` - Upload image, get cropped questions as Base64
- `GET /` - Health check

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

## 🔮 Future Enhancements

1. **Answer Matching**: Parse mark schemes and link questions to answers
2. **OCR Integration**: Extract question text using Mathpix or similar
3. **Database Integration**: Store questions and answers in structured format
4. **Multi-subject Support**: Extend to other A-Level subjects
5. **Question Classification**: Categorize by topic/difficulty

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