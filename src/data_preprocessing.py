import pandas as pd
import os

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

def load_dataset(path, label=None):
    df = pd.read_csv(path)
    df = df[['payload', 'label']].dropna()
    df['payload'] = df['payload'].astype(str).str.lower().str.strip()

    if label is not None:
        df['label'] = label
    else:
        df['label'] = df['label'].astype(int)

    return df


def main():
    print(" Loading datasets...")

    # XSS: already binary (0 = benign, 1 = XSS)
    xss = load_dataset(f"{RAW_DIR}/xss.csv")

    # SQL Injection
    sqli = load_dataset(f"{RAW_DIR}/sqli.csv", label=2)

    # Phishing
    phishing = load_dataset(f"{RAW_DIR}/phishing.csv", label=3)

    # Merge all
    full_df = pd.concat([xss, sqli, phishing], ignore_index=True)

    # Shuffle dataset
    full_df = full_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save final dataset
    output_path = f"{PROCESSED_DIR}/full_dataset.csv"
    full_df.to_csv(output_path, index=False)

    print(" Dataset preprocessing completed")
    print(" Saved to:", output_path)
    print("\n Label distribution:")
    print(full_df['label'].value_counts())

if __name__ == "__main__":
    main()
