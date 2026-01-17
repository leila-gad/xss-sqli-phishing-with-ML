import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import os

LABEL_MAP = {
    0: "Benign",
    1: "XSS",
    2: "SQL Injection",
    3: "Phishing"
}

DATA_DIR = "data/processed"
MODEL_DIR = "models"
RESULTS_DIR = "results"

def main():
    # Load test dataset
    df_test = pd.read_csv(f"{DATA_DIR}/test.csv")

    if "payload" not in df_test.columns or "label" not in df_test.columns:
        raise ValueError(" test.csv must contain 'payload' and 'label' columns")

    X_text = df_test["payload"].astype(str)
    y_test = df_test["label"].astype(int)

    # Load model & vectorizer
    model_path = f"{MODEL_DIR}/ids_model_xgb.pkl"
    vectorizer_path = f"{MODEL_DIR}/vectorizer.pkl"

    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        raise FileNotFoundError(" Model or vectorizer not found. Train the model first.")
        
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    # Vectorize test data (IMPORTANT)
    X_test = vectorizer.transform(X_text)

    # Predict
    y_pred = model.predict(X_test)
    print("DEBUG y_test unique:", sorted(y_test.unique()))
    print("DEBUG y_pred unique:", sorted(set(y_pred)))

    # Labels present in test set
    labels = sorted(np.unique(y_test))
    target_names = [LABEL_MAP[l] for l in labels]

    print("\n Classification Report:\n")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=labels,
            target_names=target_names,
            zero_division=0
        )
    )

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm)
    plt.colorbar()
    plt.xticks(range(len(target_names)), target_names, rotation=45)
    plt.yticks(range(len(target_names)), target_names)
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.savefig(f"{RESULTS_DIR}/confusion_matrix.png")
    plt.show()
    print(" Evaluation completed successfully.")
    print(" Saved to results/confusion_matrix.png")

if __name__ == "__main__":
    main()
