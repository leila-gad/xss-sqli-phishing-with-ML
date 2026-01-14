import pandas as pd

def load_and_label(path, label):
    df = pd.read_csv(path)
    df.dropna(inplace=True)
    df['payload'] = df['payload'].astype(str).str.lower().str.strip()
    df['label'] = label
    return df[['payload', 'label']]

def main():
    xss = load_and_label("data/raw/xss.csv", 1)
    sqli = load_and_label("data/raw/sqli.csv", 2)
    phishing = load_and_label("data/raw/phishing.csv", 3)

    full_df = pd.concat([xss, sqli, phishing], ignore_index=True)

    full_df.to_csv("data/processed/full_dataset.csv", index=False)
    print("Merged dataset saved.")

if __name__ == "__main__":
    main()
