import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
ARTIFACTS_DIR = Path("artifacts")


def create_candidates():
    return {
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


def evaluate_with_cross_validation(candidates, features, targets):
    cross_validator = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    results = {}

    for model_name, model in candidates.items():
        scores = cross_validate(
            model,
            features,
            targets,
            cv=cross_validator,
            scoring=["accuracy", "f1_macro"],
            n_jobs=1,
        )

        accuracy_scores = scores["test_accuracy"]
        macro_f1_scores = scores["test_f1_macro"]

        results[model_name] = {
            "accuracy_per_fold": accuracy_scores.tolist(),
            "macro_f1_per_fold": macro_f1_scores.tolist(),
            "accuracy_mean": float(accuracy_scores.mean()),
            "accuracy_std": float(accuracy_scores.std()),
            "macro_f1_mean": float(macro_f1_scores.mean()),
            "macro_f1_std": float(macro_f1_scores.std()),
        }

    return results


def main():
    features, targets = load_wine(return_X_y=True, as_frame=True)

    train_features, test_features, train_targets, test_targets = (
        train_test_split(
            features,
            targets,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=targets,
        )
    )

    candidates = create_candidates()

    cv_results = evaluate_with_cross_validation(
        candidates,
        train_features,
        train_targets,
    )

    selected_model_name = max(
        cv_results,
        key=lambda name: cv_results[name]["macro_f1_mean"],
    )

    selected_model = candidates[selected_model_name]
    selected_model.fit(train_features, train_targets)

    holdout_predictions = selected_model.predict(test_features)

    holdout_results = {
        "accuracy": float(
            accuracy_score(test_targets, holdout_predictions)
        ),
        "macro_f1": float(
            f1_score(
                test_targets,
                holdout_predictions,
                average="macro",
            )
        ),
    }

    timestamp = datetime.now(timezone.utc)
    experiment_id = timestamp.strftime("%Y%m%dT%H%M%S%fZ")

    experiment = {
        "experiment_id": experiment_id,
        "created_utc": timestamp.isoformat(),
        "dataset": "scikit-learn wine recognition dataset",
        "dataset_rows": len(features),
        "feature_count": len(features.columns),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "training_samples": len(train_targets),
        "holdout_samples": len(test_targets),
        "selection_metric": "cross_validation.macro_f1_mean",
        "selected_model": selected_model_name,
        "cross_validation": {
            "strategy": "StratifiedKFold",
            "folds": CV_FOLDS,
            "performed_on": "training_data_only",
            "models": cv_results,
        },
        "holdout_evaluation": holdout_results,
    }

    ARTIFACTS_DIR.mkdir(exist_ok=True)

    joblib.dump(
        {
            "model": selected_model,
            "feature_names": features.columns.tolist(),
            "selected_model": selected_model_name,
            "experiment_id": experiment_id,
        },
        ARTIFACTS_DIR / "model.joblib",
    )

    with open(
        ARTIFACTS_DIR / "metrics.json",
        "w",
        encoding="utf-8",
    ) as metrics_file:
        json.dump(experiment, metrics_file, indent=2)

    with open(
        ARTIFACTS_DIR / "experiment_history.jsonl",
        "a",
        encoding="utf-8",
    ) as history_file:
        history_file.write(json.dumps(experiment) + "\n")

    print(json.dumps(experiment, indent=2))


if __name__ == "__main__":
    main()