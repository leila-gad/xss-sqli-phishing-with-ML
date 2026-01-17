import pandas as pd
import os
from sklearn.model_selection import train_test_split

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

def load_attack_dataset(path, attack_label):
    df = pd.read_csv(path)
    df = df[["payload", "label"]]
    df = df[df["label"] == 1]         
    df["label"] = attack_label        
    return df

def main():
    # Load attack datasets
    xss_df = load_attack_dataset(f"{RAW_DIR}/xss.csv", 1)
    sqli_df = load_attack_dataset(f"{RAW_DIR}/sqli.csv", 2)
    phishing_df = load_attack_dataset(f"{RAW_DIR}/phishing.csv", 3)

    # Benign samples (important!)
    benign_df = pd.DataFrame({
        "payload": [
            "hello world",
            "normal request",
            "this is a test",
            "user profile page",
            "search query example"
        ],
        "label": 0
    })

    # Merge
    full_df = pd.concat(
        [benign_df, xss_df, sqli_df, phishing_df],
        ignore_index=True
    )

    # Shuffle
    full_df = full_df.sample(frac=1, random_state=42)

    # Save full dataset
    full_df.to_csv(f"{PROCESSED_DIR}/full_dataset.csv", index=False)

    # Train / Test split 
    train_df, test_df = train_test_split(
        full_df,
        test_size=0.2,
        random_state=42,
        stratify=full_df["label"]
    )

    train_df.to_csv(f"{PROCESSED_DIR}/train.csv", index=False)
    test_df.to_csv(f"{PROCESSED_DIR}/test.csv", index=False)
    print(" Preprocessing & split completed")
    print("Class distribution:")
    print(full_df["label"].value_counts())

if __name__ == "__main__":
    main()
