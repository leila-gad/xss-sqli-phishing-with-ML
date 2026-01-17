import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import xgboost as xgb
import joblib
import os

def main():
    df = pd.read_csv(
    "data/processed/full_dataset.csv",
    engine="python",
    on_bad_lines="skip"
)
    X_text = df["payload"].astype(str)
    y = df["label"].astype(int)

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=5000,  # fixed feature size
        lowercase=True
    )

    X_features = vectorizer.fit_transform(X_text)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y, test_size=0.2, random_state=42, stratify=y
    )
   
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        objective="multi:softprob",  # allows confidence/probabilities
        num_class=len(y.unique()),
        eval_metric='mlogloss'
    )

    model.fit(X_train, y_train)
    # Evaluate model
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(vectorizer, "models/vectorizer.pkl")
    joblib.dump(model, "models/ids_model_xgb.pkl")
    print("Vectorizer and XGBoost model saved successfully.")

if __name__ == "__main__":
    main()
