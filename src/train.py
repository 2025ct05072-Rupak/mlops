
from pathlib import Path
import json

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    RocCurveDisplay,
)

from src.data import load_dataset, split_features_target

ARTIFACTS = Path("artifacts")
MODEL_PATH = ARTIFACTS / "heart_disease_model.joblib"
RANDOM_STATE = 42


def build_pipeline(X, estimator):
    numeric_cols = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_cols = [
        col for col in X.columns
        if col not in numeric_cols
    ]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_cols),
        ("categorical", categorical_pipeline, categorical_cols),
    ])

    return Pipeline([
        ("preprocessing", preprocessor),
        ("model", estimator),
    ])


def train():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    df = load_dataset()
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "roc_auc": "roc_auc",
    }

    mlflow.set_experiment("heart-disease-classification")

    trained_models = {}
    comparison = []

    for name, estimator in models.items():
        pipeline = build_pipeline(X_train, estimator)

        with mlflow.start_run(run_name=name):
            mlflow.log_param("model", name)
            mlflow.log_param("random_state", RANDOM_STATE)
            mlflow.log_param("cv_folds", 5)
            mlflow.log_param("test_size", 0.20)

            scores = cross_validate(
                pipeline,
                X_train,
                y_train,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
            )

            result = {"model": name}

            for metric in scoring:
                mean_score = float(
                    np.mean(scores[f"test_{metric}"])
                )
                std_score = float(
                    np.std(scores[f"test_{metric}"])
                )

                result[f"cv_{metric}_mean"] = mean_score
                result[f"cv_{metric}_std"] = std_score

                mlflow.log_metric(
                    f"cv_{metric}_mean", mean_score
                )
                mlflow.log_metric(
                    f"cv_{metric}_std", std_score
                )

            pipeline.fit(X_train, y_train)

            trained_models[name] = pipeline
            comparison.append(result)

           

            mlflow.sklearn.log_model(
                pipeline,
                name="model",
                skops_trusted_types=[
                    "numpy.dtype",
                    "sklearn.tree._tree.Tree",
                ],
            )

            print(f"\n{name}")
            print(result)

    # Choose using validation results, not test results.
    best_result = max(
        comparison,
        key=lambda row: row["cv_roc_auc_mean"],
    )

    best_name = best_result["model"]
    best_model = trained_models[best_name]

    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    metrics = {
        "selected_model": best_name,
        "test_accuracy": float(
            accuracy_score(y_test, y_pred)
        ),
        "test_precision": float(
            precision_score(y_test, y_pred, zero_division=0)
        ),
        "test_recall": float(
            recall_score(y_test, y_pred, zero_division=0)
        ),
        "test_roc_auc": float(
            roc_auc_score(y_test, y_prob)
        ),
        "test_rows": int(len(y_test)),
    }

    joblib.dump(best_model, MODEL_PATH)

    (ARTIFACTS / "test_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    pd.DataFrame(comparison).to_csv(
        ARTIFACTS / "cv_model_comparison.csv",
        index=False,
    )

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0,
    )

    (ARTIFACTS / "classification_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        confusion_matrix(y_test, y_pred),
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # ROC curve
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_test, y_prob, ax=ax)
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "roc_curve.png", dpi=150)
    plt.close(fig)

    # Track final test results and artifacts.
    with mlflow.start_run(run_name=f"final-{best_name}"):
        mlflow.log_param("selected_model", best_name)

        for name, value in metrics.items():
            if name.startswith("test_") and isinstance(
                value, (int, float)
            ):
                mlflow.log_metric(name, value)

        for path in [
            MODEL_PATH,
            ARTIFACTS / "test_metrics.json",
            ARTIFACTS / "cv_model_comparison.csv",
            ARTIFACTS / "classification_report.json",
            ARTIFACTS / "confusion_matrix.png",
            ARTIFACTS / "roc_curve.png",
        ]:
            mlflow.log_artifact(str(path))

    print("\nFinal test metrics:")
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved model: {MODEL_PATH.resolve()}")

    return metrics


if __name__ == "__main__":
    train()
