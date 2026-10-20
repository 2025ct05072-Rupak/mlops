
from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo

DATA_DIR = Path("data")
DATA_PATH = DATA_DIR / "heart_disease.csv"


def download_dataset():
    """Download UCI Heart Disease dataset ID 45."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    heart_disease = fetch_ucirepo(id=45)
    X = heart_disease.data.features.copy()
    y = heart_disease.data.targets.copy()

    print("Dataset metadata:")
    print(heart_disease.metadata)
    print("\nVariables:")
    print(heart_disease.variables)

    target_col = "num" if "num" in y.columns else y.columns[0]
    original_target = pd.to_numeric(
        y[target_col], errors="coerce"
    )

    df = X.copy()
    df["target"] = (original_target > 0).astype(float)
    df.loc[original_target.isna(), "target"] = float("nan")

    df = df.dropna(subset=["target"]).copy()
    df["target"] = df["target"].astype(int)

    df.to_csv(DATA_PATH, index=False)

    print(f"\nSaved dataset to {DATA_PATH}")
    print("Shape:", df.shape)
    print("\nMissing values:")
    print(df.isna().sum())
    print("\nTarget distribution:")
    print(df["target"].value_counts())

    return df


def load_dataset():
    """Load the CSV or download it if absent."""
    if not DATA_PATH.exists():
        return download_dataset()

    return pd.read_csv(DATA_PATH)


def split_features_target(df):
    """Separate input features and binary target."""
    if "target" not in df.columns:
        raise ValueError("Missing target column")

    df = df.dropna(subset=["target"]).copy()
    y = pd.to_numeric(df.pop("target"), errors="raise").astype(int)

    if not set(y.unique()).issubset({0, 1}):
        raise ValueError("Target must contain only 0 and 1")

    return df, y
