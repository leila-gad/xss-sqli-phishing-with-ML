import joblib

# Consistent label mapping (SAME AS API & TRAINING)
LABEL_MAP = {
    0: "Benign",
    1: "XSS",
    2: "SQL Injection",
    3: "Phishing"
}

# Load vectorizer and model
vectorizer = joblib.load("models/vectorizer.pkl")
model = joblib.load("models/ids_model_xgb.pkl")

def predict(payload: str):
    if not payload.strip():
        return "Invalid input"

    # Clean payload (same as API)
    payload = payload.lower().strip()

    # Vectorize
    X = vectorizer.transform([payload])

    # Predict
    pred_idx = model.predict(X)[0]
    probs = model.predict_proba(X)[0]
    prediction = LABEL_MAP[pred_idx]
    confidence = probs[pred_idx]
    return {
        "payload": payload,
        "prediction": prediction,
        "confidence": round(float(confidence), 4),
        "all_probabilities": {
            LABEL_MAP[i]: round(float(p), 4)
            for i, p in enumerate(probs)
        }
    }

# Example usage
if __name__ == "__main__":
    test_payloads = [
        "hello world",
        "<script>alert('XSS')</script>",
        "185.66.9.51/module/beb4c415691679a9f31263f0f0165ca7",
        "http://paypal-login-secure.com"
    ]
    for p in test_payloads:
        result = predict(p)
        print("\nPayload:", p)
        print("Result:", result)

