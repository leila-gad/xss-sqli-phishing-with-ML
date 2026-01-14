import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Label name mapping (supports both 0-based and 1-based labels)
LABEL_MAP = {
    0: "XSS",
    1: "SQL Injection",
    2: "Phishing",
    1: "XSS",
    2: "SQL Injection",
    3: "Phishing"
}

def main():
    # Load test dataset
    df_test = pd.read_csv("data/processed/test.csv")
    df_test.rename(columns=lambda x: x.strip(), inplace=True)

    if "label" not in df_test.columns:
        raise ValueError("❌ 'label' column not found in test.csv")

    # Separate features and labels
    X_test = df_test.drop("label", axis=1)
    y_test = df_test["label"].astype(int)

    # Load trained model
    if not os.path.exists("models/ids_model_xgb.pkl"):
        raise FileNotFoundError("❌ Trained model not found. Run model_training.py first.")

    model = joblib.load("models/ids_model_xgb.pkl")

    # Predict
    y_pred = model.predict(X_test)

    # ---- DEBUG INFO (keep this) ----
    print("DEBUG y_test unique values:", sorted(y_test.unique()))
    print("DEBUG y_pred unique values:", sorted(set(y_pred)))
    print("DEBUG y_test dtype:", y_test.dtype)

    # Determine actual labels present
    labels = sorted(np.unique(y_test))

    # Build correct class names dynamically
    target_names = [LABEL_MAP.get(l, f"Class {l}") for l in labels]

    # ---- Classification Report ----
    print("\n📊 Classification Report:\n")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=labels,
            target_names=target_names,
            zero_division=0
        )
    )

    # ---- Confusion Matrix ----
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=target_names,
        yticklabels=target_names,
        cmap="Blues"
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    os.makedirs("results", exist_ok=True)
    plt.savefig("results/confusion_matrix.png")
    plt.show()

    print("✅ Evaluation completed successfully.")
    print("📁 Confusion matrix saved to results/confusion_matrix.png")

if __name__ == "__main__":
    main()
