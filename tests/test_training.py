import json

import joblib

from scripts import train


def test_training_pipeline(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "ARTIFACTS_DIR", tmp_path)

    train.main()

    metrics_path = tmp_path / "metrics.json"
    history_path = tmp_path / "experiment_history.jsonl"
    model_path = tmp_path / "model.joblib"

    assert metrics_path.exists()
    assert history_path.exists()
    assert model_path.exists()

    with open(metrics_path, encoding="utf-8") as metrics_file:
        metrics = json.load(metrics_file)

    assert metrics["cross_validation"]["folds"] == 5
    assert (
        metrics["cross_validation"]["performed_on"]
        == "training_data_only"
    )
    assert len(metrics["cross_validation"]["models"]) == 2
    assert metrics["selected_model"] in metrics["cross_validation"]["models"]

    holdout_results = metrics["holdout_evaluation"]
    assert 0.0 <= holdout_results["accuracy"] <= 1.0
    assert 0.0 <= holdout_results["macro_f1"] <= 1.0

    history_lines = history_path.read_text(encoding="utf-8").splitlines()
    assert len(history_lines) == 1

    history_entry = json.loads(history_lines[0])
    assert history_entry["experiment_id"] == metrics["experiment_id"]

    model_artifact = joblib.load(model_path)
    assert model_artifact["selected_model"] == metrics["selected_model"]
    assert model_artifact["experiment_id"] == metrics["experiment_id"]