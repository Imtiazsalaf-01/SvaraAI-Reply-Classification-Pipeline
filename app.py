from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import os
import time

# Initialize FastAPI app
app = FastAPI(
    title="SvaraAI Reply Classifier",
    description="API for classifying email reply sentiment as positive, negative, or neutral",
    version="1.0.0"
)

# Global variables for models
baseline_model = None
transformer_model = None
transformer_tokenizer = None
device = None

class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    label: str
    confidence: float

def load_models():
    """Load the best available model"""
    global baseline_model, transformer_model, transformer_tokenizer, device
    
    # Try to load transformer model first
    try:
        if os.path.exists("models/transformer"):
            transformer_tokenizer = AutoTokenizer.from_pretrained("models/transformer")
            transformer_model = AutoModelForSequenceClassification.from_pretrained("models/transformer")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            transformer_model.to(device)
            transformer_model.eval()
            print("Transformer model loaded successfully")
            return "transformer"
    except Exception as e:
        print(f"Could not load transformer model: {e}")
    
    # Fall back to baseline model
    try:
        baseline_model = joblib.load("models/baseline_model.joblib")
        print("Baseline model loaded successfully")
        return "baseline"
    except FileNotFoundError:
        print("No models found!")
        return None

def predict_with_model(text: str, model_type: str):
    """Make prediction using the specified model"""
    if model_type == "transformer" and transformer_model is not None:
        # Use transformer model
        inputs = transformer_tokenizer(
            text, truncation=True, padding=True, max_length=128, return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = transformer_model(**inputs)
            probabilities = F.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
        
        # Get label mappings
        id2label = {int(k): v for k, v in transformer_model.config.id2label.items()}
        predicted_id = int(probabilities.argmax())
        predicted_label = id2label[predicted_id]
        confidence = float(probabilities[predicted_id])
        
    elif baseline_model is not None:
        # Use baseline model
        prediction = baseline_model.predict([text])[0]
        probabilities = baseline_model.predict_proba([text])[0]
        predicted_label = prediction
        confidence = float(max(probabilities))
        
    else:
        raise HTTPException(status_code=500, detail="No model available")
    
    return predicted_label, confidence

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    global model_type
    print("Loading models...")
    model_type = load_models()
    if model_type is None:
        print("Warning: No models could be loaded!")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "SvaraAI Reply Sentiment Classifier",
        "version": "1.0.0",
        "status": "ready"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make sentiment prediction on input text
    
    Example input: {"text": "Looking forward to the demo!"}
    Example output: {"label": "positive", "confidence": 0.87}
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        # Determine which model to use (prefer transformer if available)
        use_model = "transformer" if transformer_model is not None else "baseline"
        label, confidence = predict_with_model(request.text, use_model)
        
        return PredictionResponse(label=label, confidence=confidence)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Global variable to track which model is loaded
model_type = None

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting SvaraAI Reply Classifier API")
    print("Visit http://localhost:8000/docs for API documentation")
    uvicorn.run(app, host="0.0.0.0", port=8000)