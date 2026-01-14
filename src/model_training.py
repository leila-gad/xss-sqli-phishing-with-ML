import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import xgboost as xgb
import joblib
import os

def main():
    # Read features
    df = pd.read_csv("data/processed/features.csv")
    df.rename(columns=lambda x: x.strip(), inplace=True)  # Remove extra spaces

    # Check label column exists
    if 'label' not in df.columns:
        raise ValueError("Column 'label' not found in features.csv")

    X = df.drop("label", axis=1)
    y = df["label"].astype(int)

    # Ensure labels start from 0
    y = y - min(y)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Create models folder if it doesn't exist
    os.makedirs("models", exist_ok=True)

    # XGBoost multi-class
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        objective="multi:softmax",
        num_class=len(y.unique()),
        eval_metric='mlogloss'
    )

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Save model
    joblib.dump(model, "models/ids_model_xgb.pkl")
    print("XGBoost model trained and saved.")

    # Save train/test data
    os.makedirs("data/processed", exist_ok=True)
    X_train.to_csv("data/processed/train.csv", index=False)
    X_test.to_csv("data/processed/test.csv", index=False)

if __name__ == "__main__":
    main()
