
import pandas as pd
import pytest

from src.data import split_features_target


def test_split_features_and_target():
    df = pd.DataFrame({
        "age": [45, 60, 50],
        "target": [0, 1, 0],
    })

    X, y = split_features_target(df)

    assert list(X.columns) == ["age"]
    assert y.tolist() == [0, 1, 0]


def test_missing_target_is_removed():
    df = pd.DataFrame({
        "age": [45, 60],
        "target": [0, None],
    })

    X, y = split_features_target(df)

    assert len(X) == 1
    assert y.tolist() == [0]


def test_invalid_target_is_rejected():
    df = pd.DataFrame({
        "age": [45],
        "target": [2],
    })

    with pytest.raises(ValueError):
        split_features_target(df)
