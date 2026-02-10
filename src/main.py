import pandas as pd
from pathlib import Path


def load_data():
    """
    Load raw credit card churn dataset.
    """
    data_path = Path("data/raw/BankChurners.csv")
    df = pd.read_csv(data_path)
    return df


def main():
    df = load_data()
    print("Dataset loaded successfully")
    print("Shape:", df.shape)
    print("\nColumns:")
    print(df.columns)


if __name__ == "__main__":
    main()
