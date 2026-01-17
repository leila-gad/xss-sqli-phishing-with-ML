import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import xgboost as xgb
import joblib
import os

DATA_PATH = "data/processed/full_dataset.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def main():
    # Load dataset
    df = pd.read_csv(DATA_PATH)
    X_text = df["payload"].astype(str)
    y = df["label"].astype(int)

    # TF-IDF
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=5000,
        lowercase=True
    )
    X = vectorizer.fit_transform(X_text)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

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
    print(classification_report(
        y_test,
        y_pred,
        target_names=["Benign", "XSS", "SQLi", "Phishing"]
    ))

    # Save
    joblib.dump(vectorizer, f"{MODEL_DIR}/vectorizer.pkl")
    joblib.dump(model, f"{MODEL_DIR}/ids_model_xgb.pkl")
    print(" Model and vectorizer saved successfully.")

if __name__ == "__main__":
    main()
