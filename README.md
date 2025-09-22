# SvaraAI-Reply-Classification-Pipeline

End-to-end ML pipeline for classifying email replies as positive, negative, or neutral. Implements baseline (Logistic Regression) and transformer (DistilBERT) models, evaluated with Accuracy & F1, and deployed via FastAPI /predict API with Docker support.

## 🎯 Project Overview

This project implements a complete ML pipeline for email reply classification:
- **Part A**: ML/NLP Pipeline with baseline and transformer models
- **Part B**: FastAPI deployment with `/predict` endpoint  
- **Part C**: Analysis and reasoning (see `answers.md`)

## 📊 Dataset

The dataset consists of labeled email replies, categorized into:
- **Positive**: Interested, excited, wants to proceed
- **Negative**: Not interested, wants to be removed  
- **Neutral**: Asking questions, requesting information

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip

### Installation & Setup

1. **Install dependencies**
```bash
pip install -r requirements.txt
```

2. **Run the complete ML pipeline**
```bash
python notebook.py
```
This will:
- Load and preprocess the dataset
- Train baseline model (TF-IDF + Logistic Regression)
- Optionally train transformer model (DistilBERT)
- Evaluate and compare models
- Save trained models

3. **Start the API server**
```bash
python src/app.py
```
The API will be available at `http://localhost:8000`

4. **Test the API**
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "Looking forward to the demo!"}'
```

Expected response:
```json
{
  "label": "positive",
  "confidence": 0.87
}
```

## 📁 Project Structure

```
├── notebook.py                    # Main ML pipeline (Part A)
├── src/
│   └── app.py                     # FastAPI application (Part B)
├── data/
│   ├── reply_classification_dataset.csv
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
├── models/
│   ├── baseline_model.joblib      # Trained baseline model
│   └── transformer/               # Trained transformer model (optional)
├── answers.md                     # Short answers (Part C)
├── requirements.txt
├── Dockerfile
└── README.md
```

## 🔧 API Usage

### Endpoint: POST /predict

**Input:**
```json
{
  "text": "Looking forward to the demo!"
}
```

**Output:**
```json
{
  "label": "positive",
  "confidence": 0.87
}
```

### Python Example
```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={"text": "This looks great, let's schedule a call!"}
)
print(response.json())
# Output: {"label": "positive", "confidence": 0.92}
```

## 📈 Model Performance

### Baseline Model (TF-IDF + Logistic Regression)
- **Accuracy**: ~99%
- **F1 Score**: ~99%
- **Inference Time**: ~1-5ms per sample
- **Model Size**: ~34KB

### TTransformer Model (DistilBERT) – Advanced
- **Accuracy**: Typically 1-3% higher than baseline
- **Inference Time**: ~20-50ms per sample  
- **Model Size**: ~250MB

## 🐳 Docker Deployment

```bash
# Build the image
docker build -t reply-classifier .

# Run the container
docker run -p 8000:8000 reply-classifier
```

## 🛠️ Development

### Running Individual Components

```bash
# Run only the ML pipeline
python notebook.py

# Run only the API
python src/app.py

# Check API documentation
# Visit http://localhost:8000/docs
```

### Model Selection

The API automatically uses the best available model:
1. Transformer model (if available in `models/transformer/`)
2. Baseline model (fallback)

## 📋 Assignment Deliverables

✅ **Part A - ML Pipeline**: `notebook.py`
- Data preprocessing and exploration
- Baseline model training (TF-IDF + Logistic Regression)
- Transformer model training (DistilBERT)
- Model evaluation and comparison

✅ **Part B - API Deployment**: `src/app.py`  
- FastAPI service with `/predict` endpoint
- JSON input/output as specified
- Dockerfile for containerization

✅ **Part C - Analysis**: `answers.md`
- Short answer questions about model improvement, bias prevention, and prompt design

## 🚨 Troubleshooting

**Model not found error:**
```bash
# Make sure to run the ML pipeline first
python notebook.py
```

**API not starting:**
```bash
# Check if port 8000 is available
python src/app.py
```

**Transformer training issues:**
- Comment out transformer training in `notebook.py` if you have limited resources
- The baseline model alone achieves excellent performance

## 📚 Key Features

- **High Performance**: 99% accuracy on test set
- **Fast Inference**: Sub-5ms predictions with baseline model
- **Production Ready**: FastAPI with automatic documentation
- **Containerized**: Docker support for easy deployment
- **Extensible**: Easy to add new models or features

---

**📌 Assignment Alignment: Fully implements all deliverables (Part A, Part B, Part C).  
**Estimated Runtime**: 5-10 minutes for full pipeline  
**API Documentation**: Available at `http://localhost:8000/docs` when running
