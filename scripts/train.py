import json
from pathlib import Path

import joblib
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
ARTIFACTS_DIR = Path("artifacts")


def main():
    features, targets = load_wine(return_X_y=True, as_frame=True)

    train_features, test_features, train_targets, test_targets = (
        train_test_split(
            features,
            targets,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=targets,
        )
    )

    candidates = {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=5000,
                random_state=RANDOM_STATE,
            ),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
        ),
    }

    results = {}
    best_model = None
    best_model_name = None
    best_f1_score = -1.0

    for model_name, model in candidates.items():
        model.fit(train_features, train_targets)
        predictions = model.predict(test_features)

        scores = {
            "accuracy": accuracy_score(test_targets, predictions),
            "macro_f1": f1_score(
                test_targets,
                predictions,
                average="macro",
            ),
        }
        results[model_name] = scores

        if scores["macro_f1"] > best_f1_score:
            best_model = model
            best_model_name = model_name
            best_f1_score = scores["macro_f1"]

    ARTIFACTS_DIR.mkdir(exist_ok=True)

    joblib.dump(
        {
            "model": best_model,
            "feature_names": features.columns.tolist(),
        },
        ARTIFACTS_DIR / "model.joblib",
    )

    experiment = {
        "dataset": "scikit-learn wine recognition dataset",
        "random_state": RANDOM_STATE,
        "test_size": 0.2,
        "selection_metric": "macro_f1",
        "selected_model": best_model_name,
        "results": results,
    }

    with open(
        ARTIFACTS_DIR / "metrics.json",
        "w",
        encoding="utf-8",
    ) as metrics_file:
        json.dump(experiment, metrics_file, indent=2)

    print(json.dumps(experiment, indent=2))


if __name__ == "__main__":
    main()