from fastapi.testclient import TestClient

from app.main import app


VALID_FEATURES = [
    14.23,
    1.71,
    2.43,
    15.6,
    127,
    2.8,
    3.06,
    0.28,
    2.29,
    5.64,
    1.04,
    3.92,
    1065,
]


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_loaded": True,
    }


def test_valid_prediction():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"features": VALID_FEATURES},
        )

    assert response.status_code == 200

    result = response.json()
    assert result["predicted_class"] in [0, 1, 2]
    assert 0.0 <= result["confidence"] <= 1.0


def test_too_few_features_are_rejected():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"features": [1.0] * 12},
        )

    assert response.status_code == 422


def test_too_many_features_are_rejected():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"features": [1.0] * 14},
        )

    assert response.status_code == 422


def test_non_numeric_features_are_rejected():
    invalid_features = VALID_FEATURES.copy()
    invalid_features[0] = "not-a-number"

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"features": invalid_features},
        )

    assert response.status_code == 422