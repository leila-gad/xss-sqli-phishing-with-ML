import pandas as pd
import os
from sklearn.model_selection import train_test_split

def extract_features(payload):
    # HARD SAFETY
    if payload is None or pd.isna(payload):
        payload = ""
    else:
        payload = str(payload)

    features = {
        "length": len(payload),
        "digit_count": sum(c.isdigit() for c in payload),
        "special_char_count": sum(not c.isalnum() for c in payload),
        "lt_count": payload.count("<"),
        "gt_count": payload.count(">"),
        "script_count": payload.lower().count("script"),
        "alert_count": payload.lower().count("alert"),
        "onerror_count": payload.lower().count("onerror"),
        "sql_keywords": sum(payload.lower().count(k) for k in
                            ["select", "union", "insert", "drop", "or", "and"]),
        "url_count": payload.lower().count("http") + payload.lower().count("www")
    }

    return features


def main():
    print("🔹 Loading dataset...")
    df = pd.read_csv("data/processed/full_dataset.csv")

    print("🔹 Raw NaN check:")
    print(df.isna().sum())

    # FORCE CLEAN
    df["payload"] = df["payload"].fillna("").astype(str)
    df["label"] = df["label"].fillna(-1).astype(int)

    print("🔹 After cleaning:")
    print(df.isna().sum())

    # Extract features
    print("🔹 Extracting features...")
    features = df["payload"].apply(extract_features)
    X = pd.DataFrame(features.tolist())

    print("🔹 Feature NaN check:")
    print(X.isna().sum())

    # FINAL CLEAN
    X = X.fillna(0)
    X = X.replace([float("inf"), float("-inf")], 0)

    # Add label
    X["label"] = df["label"]

    print("🔹 Final NaN check (MUST be 0):")
    print(X.isna().sum())

    # HARD ASSERT — WILL CRASH IF ANY NaN EXISTS
    assert not X.isna().any().any(), "❌ NaNs STILL EXIST IN FEATURES"

    os.makedirs("data/processed", exist_ok=True)
    X.to_csv("data/processed/features.csv", index=False)

    print("✅ Feature extraction SUCCESS — no NaNs")

    # Train / Test split
    X_train, X_test = train_test_split(
        X,
        test_size=0.2,
        random_state=42
    )

    X_train.to_csv("data/processed/train.csv", index=False)
    X_test.to_csv("data/processed/test.csv", index=False)

    print("✅ Train/test saved")

if __name__ == "__main__":
    main()
