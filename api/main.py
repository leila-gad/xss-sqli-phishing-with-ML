
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import os
import json
import joblib

app = FastAPI(
    title="ML-Based Intrusion Detection System",
    description="Real-time detection of XSS, SQLi, and Phishing attacks",
    version="1.0.0"
)

# Enable CORS so frontend can access API from any port
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
vectorizer = None

LABEL_MAP = {
    0: "Benign",
    1: "XSS",
    2: "SQLi",
    3: "Phishing"
}

# Logs
os.makedirs("logs", exist_ok=True)
LOG_FILE = "logs/ids_logs.json"

def load_models():
    global model, vectorizer
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        MODEL_DIR = os.path.join(BASE_DIR, "..", "models")  # adjust if models are elsewhere
        vectorizer = joblib.load(os.path.join(MODEL_DIR, "vectorizer.pkl"))
        model = joblib.load(os.path.join(MODEL_DIR, "ids_model_xgb.pkl"))
        print(" Models loaded successfully")
        return True
    except FileNotFoundError as e:
        print(" Model file not found:", e)
        return False
    except Exception as e:
        print(" Unexpected error:", e)
        return False

def log_detection(payload: str, prediction: str, confidence: float, severity: str):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "payload": payload,
        "attack_type": prediction,
        "confidence": float(confidence),
        "severity": severity
    }
    logs = []
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r") as f:
                logs = json.load(f)
        except json.JSONDecodeError:
            logs = []
    logs.append(log_entry)
    logs = logs[-1000:]  # keep last 1000 logs
    with open(LOG_FILE, "w") as f:
        json.dump(logs, f, indent=2)

@app.on_event("startup")
async def startup_event():
    success = load_models()
    if not success:
        print(" Warning: Models not loaded. /predict endpoint will not work.")

# Request / Response Models
class PredictionRequest(BaseModel):
    payload: str

class PredictionResponse(BaseModel):
    payload: str
    prediction: str
    confidence: float
    all_probabilities: dict
    severity: str
    timestamp: str
    message: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    vectorizer_loaded: bool
    timestamp: str

# Endpoints
@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "ML-Based Intrusion Detection System API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)",
            "logs": "/logs",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    return {
        "status": "healthy" if (model is not None and vectorizer is not None) else "unhealthy",
        "model_loaded": model is not None,
        "vectorizer_loaded": vectorizer is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_attack(request: PredictionRequest):
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Models not loaded. Please train the model first.")
    if not request.payload.strip():
        raise HTTPException(status_code=400, detail="Payload cannot be empty")
    
    try:
        cleaned_payload = [request.payload.lower().strip()]
        X = vectorizer.transform(cleaned_payload)
        
        # Prediction
        prediction_idx = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
        predicted_class = LABEL_MAP.get(prediction_idx, f"Class_{prediction_idx}")
        confidence = float(probabilities[prediction_idx])

        # Severity
        if predicted_class == "Benign":
            severity = "low"
        elif confidence < 0.8:
            severity = "medium"
        else:
            severity = "high"
        all_probs = {LABEL_MAP.get(i, f"Class_{i}"): float(prob) for i, prob in enumerate(probabilities)}
        if predicted_class != "Benign":
            log_detection(request.payload, predicted_class, confidence, severity)

        return {
            "payload": request.payload,
            "prediction": predicted_class,
            "confidence": confidence,
            "all_probabilities": all_probs,
            "severity": severity,
            "timestamp": datetime.now().isoformat(),
            "message": f"Detected {predicted_class} attack with {confidence*100:.2f}% confidence"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/logs", tags=["Logs"])
async def get_logs(limit: int = 50):
    if not os.path.exists(LOG_FILE):
        return {"logs": [], "count": 0}
    try:
        with open(LOG_FILE, "r") as f:
            logs = json.load(f)
        limit = min(limit, 1000)
        recent_logs = logs[-limit:]
        return {"logs": recent_logs[::-1], "count": len(recent_logs), "total_logs": len(logs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading logs: {str(e)}")

@app.delete("/logs", tags=["Logs"])
async def clear_logs():
    try:
        if os.path.exists(LOG_FILE):
            os.remove(LOG_FILE)
        return {"message": "Logs cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing logs: {str(e)}")


# Run API
if __name__ == "__main__":
    import uvicorn
    print("="*70)
    print("Starting ML-Based Intrusion Detection System API")
    print("="*70)
    uvicorn.run(app, host="0.0.0.0", port=8000)

