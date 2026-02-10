import pandas as pd
from pathlib import Path


def load_raw_data():
    data_path = Path("data/raw/BankChurners.csv")
    return pd.read_csv(data_path)


def preprocess_data(df):
    # Drop identifier column
    if "CLIENTNUM" in df.columns:
        df = df.drop(columns=["CLIENTNUM"])

    # Encode target variable
    df["Attrition_Flag"] = df["Attrition_Flag"].map({
        "Existing Customer": 0,
        "Attrited Customer": 1
    })

    return df


def save_processed_data(df):
    output_path = Path("data/processed")
    output_path.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path / "clean_churn_data.csv", index=False)


def main():
    df = load_raw_data()
    df_clean = preprocess_data(df)
    save_processed_data(df_clean)

    print("Clean modeling dataset created successfully")
    print("Shape:", df_clean.shape)
    print("Target distribution:")
    print(df_clean["Attrition_Flag"].value_counts())


if __name__ == "__main__":
    main()
