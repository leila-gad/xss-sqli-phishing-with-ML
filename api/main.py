"""
ML-Based Intrusion Detection System - FastAPI Deployment
Real-time detection of XSS, SQL Injection, and Phishing attacks
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import os
import json
import joblib

# Initialize FastAPI
app = FastAPI(
    title="ML-Based Intrusion Detection System",
    description="Real-time detection of XSS, SQLi, and Phishing attacks",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
vectorizer = None
label_names = {
    0: "Benign",
    1: "XSS",
    2: "SQLi",
    3: "Phishing"
}

# Logs directory
os.makedirs('logs', exist_ok=True)
LOG_FILE = 'logs/ids_logs.json'

# Load models function
def load_models():
    """Load trained model and vectorizer (robust paths using joblib)"""
    global model, vectorizer

    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        MODEL_DIR = os.path.join(BASE_DIR, "..", "models")

        vectorizer_path = os.path.join(MODEL_DIR, "vectorizer.pkl")
        model_path = os.path.join(MODEL_DIR, "ids_model_xgb.pkl")

        print("Loading vectorizer from:", vectorizer_path)
        print("Loading model from:", model_path)

        vectorizer = joblib.load(vectorizer_path)
        model = joblib.load(model_path)

        print("✓ Models loaded successfully")
        return True

    except FileNotFoundError as e:
        print("✗ File not found:", e)
        return False

    except Exception as e:
        print("✗ Unexpected error:", e)
        return False

# ---------------------------
# 4️⃣ Log detection
# ---------------------------
def log_detection(payload: str, prediction: str, confidence: float, severity: str):
    """Log detected attacks to JSON file"""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "payload": payload,
        "attack_type": prediction,
        "confidence": float(confidence),
        "severity": severity
    }

    # Load existing logs
    logs = []
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, 'r') as f:
                logs = json.load(f)
        except json.JSONDecodeError:
            logs = []

    # Append new log and keep last 1000
    logs.append(log_entry)
    logs = logs[-1000:]

    # Save
    with open(LOG_FILE, 'w') as f:
        json.dump(logs, f, indent=2)

# ---------------------------
# 5️⃣ Startup event
# ---------------------------
@app.on_event("startup")
async def startup_event():
    success = load_models()
    if not success:
        print("WARNING: Models not loaded. /predict endpoint will not work.")

# ---------------------------
# 6️⃣ Request/Response Models

class PredictionRequest(BaseModel):
    payload: str

    class Config:
        json_schema_extra = {
            "example": {
                "payload": "<script>alert('XSS')</script>"
            }
        }

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

# ---------------------------
# 7️⃣ Root & Health endpoints
# ---------------------------
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

# ---------------------------
# 8️⃣ Prediction endpoint
# ---------------------------
@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_attack(request: PredictionRequest):
    if model is None or vectorizer is None:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Please ensure train_model.py has been run."
        )

    if not request.payload or not request.payload.strip():
        raise HTTPException(status_code=400, detail="Payload cannot be empty")

    try:
        # Preprocess payload
        cleaned_payload = request.payload.lower().strip()
        X = vectorizer.transform([cleaned_payload])

        # Predict
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]

        predicted_class = label_names.get(prediction, f"Class_{prediction}")
        confidence = float(probabilities[prediction])

        # ---------------------------
        # ✅ Corrected Severity Logic
        # ---------------------------
        if predicted_class == "Benign":
            severity = "low"
        elif confidence < 0.8:
            severity = "medium"
        else:
            severity = "high"

        # Probabilities dict
        all_probs = {
            label_names.get(i, f"Class_{i}"): float(prob) 
            for i, prob in enumerate(probabilities)
        }

        # Log attacks (skip benign)
        if predicted_class != "Benign":
            log_detection(
                payload=request.payload,
                prediction=predicted_class,
                confidence=confidence,
                severity=severity
            )

        return {
            "payload": request.payload,
            "prediction": predicted_class,
            "confidence": confidence,
            "all_probabilities": all_probs,
            "severity": severity,
            "timestamp": datetime.now().isoformat(),
            "message": f"Detected {predicted_class} attack with {confidence*100:.2f}% confidence"
                    if predicted_class != "Benign"
                    else "No malicious activity detected"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# ---------------------------
# 9️⃣ Logs endpoints
# ---------------------------
@app.get("/logs", tags=["Logs"])
async def get_logs(limit: int = 50):
    if not os.path.exists(LOG_FILE):
        return {"logs": [], "count": 0}

    try:
        with open(LOG_FILE, 'r') as f:
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

# ---------------------------
# 10️⃣ Run API
# ---------------------------
if __name__ == "__main__":
    import uvicorn
    print("=" * 70)
    print("Starting ML-Based Intrusion Detection System API")
    print("=" * 70)
    uvicorn.run(app, host="0.0.0.0", port=8000)
