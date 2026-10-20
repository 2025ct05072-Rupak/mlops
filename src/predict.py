
from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path("artifacts/heart_disease_model.joblib")


def load_model(path=MODEL_PATH):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Model not found: {path}. Run python -m src.train first."
        )

    return joblib.load(path)


def predict_one(payload, model=None):
    if model is None:
        model = load_model()

    # One JSON object becomes one input row.
    row = pd.DataFrame([payload])

    prediction = int(model.predict(row)[0])
    probabilities = model.predict_proba(row)[0]

    return {
        "prediction": prediction,
        "label": (
            "disease_present"
            if prediction == 1
            else "no_disease_detected"
        ),
        "confidence": float(max(probabilities)),
        "probability_disease_present": float(probabilities[1]),
        "note": "Educational output, not a medical diagnosis.",
    }
