
import pandas as pd

from sklearn.linear_model import LogisticRegression
from src.train import build_pipeline


def test_pipeline_handles_missing_values():
    X = pd.DataFrame({
        "age": [40, 50, 60, 35, None, 70, 45, 55],
        "sex": ["M", "F", "M", "F", "M", "F", "M", "F"],
    })

    y = [0, 1, 1, 0, 0, 1, 0, 1]

    model = build_pipeline(
        X,
        LogisticRegression(max_iter=500),
    )

    model.fit(X, y)

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    assert len(predictions) == len(y)
    assert set(predictions).issubset({0, 1})
    assert probabilities.shape == (len(y), 2)
