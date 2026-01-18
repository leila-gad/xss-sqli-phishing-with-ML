from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import joblib
import os
import json

app = FastAPI(
    title="ML-Based Intrusion Detection System",
    description="Detection of XSS, SQL Injection, and Phishing attacks using Machine Learning",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
vectorizer = None
LABEL_MAP = {0: "Benign", 1: "XSS", 2: "SQL Injection", 3: "Phishing"}
LOG_DIR = "logs"
LOG_FILE = f"{LOG_DIR}/ids_logs.json"
os.makedirs(LOG_DIR, exist_ok=True)

def load_models():
    global model, vectorizer
    try:
        model = joblib.load("models/ids_model_xgb.pkl")
        vectorizer = joblib.load("models/vectorizer.pkl")
        print(" Model and vectorizer loaded successfully")
    except Exception as e:
        print(" Failed to load model/vectorizer:", e)

@app.on_event("startup")
async def startup_event():
    load_models()

class PredictionRequest(BaseModel):
    payload: str

class PredictionResponse(BaseModel):
    payload: str
    prediction: str
    confidence: float
    probabilities: dict
    severity: str
    timestamp: str

def log_detection(payload, prediction, confidence, severity):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "payload": payload,
        "prediction": prediction,
        "confidence": confidence,
        "severity": severity
    }

    logs = []
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r") as f:
                logs = json.load(f)
        except json.JSONDecodeError:
            logs = []

    logs.append(entry)
    logs = logs[-1000:]  # Keep last 1000 logs
    with open(LOG_FILE, "w") as f:
        json.dump(logs, f, indent=2)

@app.get("/")
def root():
    return {"message": "ML-Based IDS API is running", "docs": "/docs", "health": "/health"}

@app.get("/health")
def health():
    return {
        "model_loaded": model is not None,
        "vectorizer_loaded": vectorizer is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not request.payload.strip():
        raise HTTPException(status_code=400, detail="Payload cannot be empty")

    payload = request.payload.strip()  
    X = vectorizer.transform([payload])
    pred_idx = int(model.predict(X)[0])
    probs = model.predict_proba(X)[0]
    prediction = LABEL_MAP[pred_idx]
    confidence = float(probs[pred_idx])
    probabilities = {LABEL_MAP[i]: round(float(p), 4) for i, p in enumerate(probs)}

    if prediction == "Benign":
        severity = "low"
    elif confidence < 0.8:
        severity = "medium"
    else:
        severity = "high"

    if prediction != "Benign":
        log_detection(payload, prediction, confidence, severity)

    return {
        "payload": request.payload,
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "probabilities": probabilities,
        "severity": severity,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/logs")
def get_logs(limit: int = 50):
    if not os.path.exists(LOG_FILE):
        return {"count": 0, "logs": []}
    with open(LOG_FILE, "r") as f:
        logs = json.load(f)
    return {"count": min(limit, len(logs)), "logs": logs[-limit:][::-1]}

@app.delete("/logs")
def clear_logs():
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
    return {"message": "Logs cleared"}
