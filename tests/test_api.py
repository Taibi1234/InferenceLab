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

def test_batch_prediction():
    with TestClient(app) as client:
        response = client.post(
            "/predict/batch",
            json={"rows": [VALID_FEATURES, VALID_FEATURES]},
        )

    assert response.status_code == 200

    result = response.json()
    assert result["count"] == 2
    assert len(result["predictions"]) == 2

    for prediction in result["predictions"]:
        assert prediction["predicted_class"] in [0, 1, 2]
        assert 0.0 <= prediction["confidence"] <= 1.0


def test_empty_batch_is_rejected():
    with TestClient(app) as client:
        response = client.post(
            "/predict/batch",
            json={"rows": []},
        )

    assert response.status_code == 422


def test_invalid_row_inside_batch_is_rejected():
    with TestClient(app) as client:
        response = client.post(
            "/predict/batch",
            json={"rows": [VALID_FEATURES, [1.0] * 12]},
        )

    assert response.status_code == 422


def test_batch_size_limit_is_enforced():
    with TestClient(app) as client:
        response = client.post(
            "/predict/batch",
            json={"rows": [VALID_FEATURES] * 1001},
        )

    assert response.status_code == 422

def test_metrics_endpoint():
    with TestClient(app) as client:
        client.get("/health")
        response = client.get("/metrics")

    assert response.status_code == 200
    assert "inferencelab_http_requests_total" in response.text
    assert "inferencelab_http_request_duration_seconds" in response.text

def test_drift_check():
    with TestClient(app) as client:
        response = client.post(
            "/drift/check",
            json={"rows": [VALID_FEATURES] * 30},
        )

    assert response.status_code == 200

    result = response.json()
    assert result["sample_count"] == 30
    assert result["overall_status"] in [
        "no_significant_drift",
        "moderate_drift",
        "significant_drift",
    ]
    assert len(result["features"]) == 13


def test_small_drift_sample_is_rejected():
    with TestClient(app) as client:
        response = client.post(
            "/drift/check",
            json={"rows": [VALID_FEATURES] * 29},
        )

    assert response.status_code == 422


def test_extreme_batch_reports_significant_drift():
    extreme_rows = [[999.0] * 13] * 30

    with TestClient(app) as client:
        response = client.post(
            "/drift/check",
            json={"rows": extreme_rows},
        )

    assert response.status_code == 200
    assert response.json()["overall_status"] == "significant_drift"