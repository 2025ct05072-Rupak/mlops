
from fastapi.testclient import TestClient
import src.api as api


class FakeModel:
    def predict(self, frame):
        return [1]

    def predict_proba(self, frame):
        return [[0.2, 0.8]]


def test_health(monkeypatch):
    with TestClient(api.app) as client:
        monkeypatch.setattr(api, "MODEL", FakeModel())

        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_prediction(monkeypatch):
    with TestClient(api.app) as client:
        monkeypatch.setattr(api, "MODEL", FakeModel())

        response = client.post(
            "/predict",
            json={"age": 55, "sex": 1},
        )

    assert response.status_code == 200
    assert response.json()["prediction"] == 1
    assert response.json()["probability_disease_present"] == 0.8


def test_prediction_without_model(monkeypatch):
    with TestClient(api.app) as client:
        monkeypatch.setattr(api, "MODEL", None)

        response = client.post(
            "/predict",
            json={"age": 55},
        )

    assert response.status_code == 503
    assert response.json()["detail"] == "Model is not loaded"
