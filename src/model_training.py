import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import joblib
import os

DATA_DIR = "data/processed"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
LABEL_NAMES = ["Benign", "XSS", "SQLi", "Phishing"]

def main():
    # Load train & test data
    train_df = pd.read_csv(f"{DATA_DIR}/train.csv")
    test_df = pd.read_csv(f"{DATA_DIR}/test.csv")
    X_train_text = train_df["payload"].astype(str)
    y_train = train_df["label"].astype(int)
    X_test_text = test_df["payload"].astype(str)
    y_test = test_df["label"].astype(int)

    # TF-IDF (FIT ONLY ON TRAIN)
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=5000,
        lowercase=True
    )
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    # Model
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        objective="multi:softprob",
        num_class=4,
        eval_metric="mlogloss"
    )
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=LABEL_NAMES
    ))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save model
    joblib.dump(vectorizer, f"{MODEL_DIR}/vectorizer.pkl")
    joblib.dump(model, f"{MODEL_DIR}/ids_model_xgb.pkl")
    print("\n Model & vectorizer saved.")

if __name__ == "__main__":
    main()
