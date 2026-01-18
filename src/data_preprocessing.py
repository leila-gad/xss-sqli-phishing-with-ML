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
    xss_df = load_attack_dataset(f"{RAW_DIR}/xss.csv", 1)
    sqli_df = load_attack_dataset(f"{RAW_DIR}/sqli.csv", 2)
    phishing_df = load_attack_dataset(f"{RAW_DIR}/phishing.csv", 3)

    benign_xss = pd.read_csv(f"{RAW_DIR}/xss.csv")
    benign_sqli = pd.read_csv(f"{RAW_DIR}/sqli.csv")
    benign_phish = pd.read_csv(f"{RAW_DIR}/phishing.csv")

    benign_df = pd.concat(
        [
            benign_xss[benign_xss["label"] == 0][["payload", "label"]],
            benign_sqli[benign_sqli["label"] == 0][["payload", "label"]],
            benign_phish[benign_phish["label"] == 0][["payload", "label"]],
        ],
        ignore_index=True
    )

    full_df = pd.concat([benign_df, xss_df, sqli_df, phishing_df], ignore_index=True)
    full_df = full_df.sample(frac=1, random_state=42)
    full_df.to_csv(f"{PROCESSED_DIR}/full_dataset.csv", index=False)
    train_df, test_df = train_test_split(
        full_df,
        test_size=0.2,
        random_state=42,
        stratify=full_df["label"]
    )

    train_df.to_csv(f"{PROCESSED_DIR}/train.csv", index=False)
    test_df.to_csv(f"{PROCESSED_DIR}/test.csv", index=False)
    print("Preprocessing & split completed")
    print("Class distribution in full dataset:")
    print(full_df["label"].value_counts())

if __name__ == "__main__":
    main()
