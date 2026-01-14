import pandas as pd
import joblib
from feature_extraction import extract_features  # Make sure this exists

# Map numeric labels back to attack names
LABELS = {
    0: "XSS",
    1: "SQL Injection",
    2: "Phishing"
}

# Load trained model
model = joblib.load("models/ids_model_xgb.pkl")

def predict(payload):
    # Convert payload to features
    features = extract_features(payload)
    df = pd.DataFrame([features])
    
    # Predict label
    pred_label = model.predict(df)[0]
    
    # Map to attack name
    return LABELS[pred_label]

# Example usage
if __name__ == "__main__":
    test_payload = "<script>alert('XSS')</script>"
    result = predict(test_payload)
    print("Payload:", test_payload)
    print("Predicted Attack:", result)
