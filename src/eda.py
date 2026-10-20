
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.data import load_dataset

OUTPUT_DIR = Path("artifacts/eda")


def run_eda():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset()

    # Summary statistics
    df.describe(include="all").transpose().to_csv(
        OUTPUT_DIR / "summary.csv"
    )

    # Missing-value report
    df.isna().sum().to_csv(
        OUTPUT_DIR / "missing_values.csv",
        header=["missing_count"],
    )

    # Histograms
    for col in df.select_dtypes(include="number").columns:
        plt.figure(figsize=(7, 4))
        sns.histplot(data=df, x=col, kde=False)
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"hist_{col}.png", dpi=150)
        plt.close()

    # Correlation heatmap
    numeric_df = df.select_dtypes(include="number")

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        numeric_df.corr(),
        cmap="coolwarm",
        center=0,
        annot=False,
    )
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "correlation_heatmap.png",
        dpi=150,
    )
    plt.close()

    # Class balance
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="target")
    plt.title("Heart Disease Class Balance")
    plt.xlabel("Target (0 = absent, 1 = present)")
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "class_balance.png",
        dpi=150,
    )
    plt.close()

    print(f"EDA results saved in {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    run_eda()
