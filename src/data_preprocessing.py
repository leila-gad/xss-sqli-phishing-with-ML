import pandas as pd
import os

# Paths
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

def load_and_process(file_path, attack_label):
    df = pd.read_csv(file_path)
    df = df[["payload", "label"]]

    # Keep only malicious samples (label == 1)
    df = df[df["label"] == 1]

    # Assign attack type
    df["label"] = attack_label
    return df

def main():
    # Load attack datasets
    xss_df = load_and_process(f"{RAW_DIR}/xss.csv", 1)       # XSS
    sqli_df = load_and_process(f"{RAW_DIR}/sqli.csv", 2)    # SQLi
    phishing_df = load_and_process(f"{RAW_DIR}/phishing.csv", 3)  # Phishing

    # Optional benign samples
    benign_df = pd.DataFrame({
        "payload": [
            "hello world",
            "test message",
            "this is a normal request",
            "sample input text"
        ],
        "label": 0
    })

    # Merge all
    full_df = pd.concat([benign_df, xss_df, sqli_df, phishing_df], ignore_index=True)

    # Shuffle
    full_df = full_df.sample(frac=1, random_state=42)

    # Save
    full_df.to_csv(f"{PROCESSED_DIR}/full_dataset.csv", index=False)
    print(" Preprocessing completed.")
    print(full_df["label"].value_counts())

if __name__ == "__main__":
    main()
