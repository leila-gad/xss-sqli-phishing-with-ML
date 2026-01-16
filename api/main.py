
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import json
from datetime import datetime
import os

# Load model & vectorizer
model = joblib.load("models/ids_model_xgb.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# Label mapping
LABELS = {
    0: "XSS",
    1: "SQL Injection",
    2: "Phishing"
}
LOG_FILE = "logs/ids_logs.json"
def log_attack(payload, attack_type, confidence):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "payload": payload,
        "attack_type": attack_type,
        "confidence": confidence,
        "severity": "HIGH" if confidence > 0.85 else "MEDIUM"
    }

    os.makedirs("logs", exist_ok=True)

    # Read existing logs
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            logs = json.load(f)
    else:
        logs = []

    logs.append(log_entry)

    # Write back to file
    with open(LOG_FILE, "w") as f:
        json.dump(logs, f, indent=4)


app = FastAPI(title="ML-Based IDS API")

class PayloadRequest(BaseModel):
    payload: str

@app.get("/health")
def health_check():
    return {"status": "OK"}

@app.post("/predict")
def predict_attack(request: PayloadRequest):
    payload = request.payload.lower().strip()
    features = vectorizer.transform([payload])
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    confidence = float(max(model.predict_proba(features)[0]))
    attack_type = LABELS[prediction]

    #  LOG THE ATTACK
    log_attack(payload, attack_type, round(confidence, 3))

    return {
        "attack_type": attack_type,
        "confidence": round(confidence, 3)
    }

@app.get("/")
def root():
    return {"message": "IDS API is running"}
