from pathlib import Path

import joblib
import pandas as pd


class ModelService:
    def __init__(self, artifact_path: Path):
        self.artifact_path = artifact_path
        self.model = None
        self.feature_names: list[str] = []

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        if not self.artifact_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at {self.artifact_path}. "
                "Run 'python scripts/train.py' first."
            )

        artifact = joblib.load(self.artifact_path)
        self.model = artifact["model"]
        self.feature_names = artifact["feature_names"]

    def predict(self, features: list[float]) -> tuple[int, float]:
        return self.predict_batch([features])[0]

    def predict_batch(
        self,
        rows: list[list[float]],
    ) -> list[tuple[int, float]]:
        if self.model is None:
            raise RuntimeError("The model has not been loaded.")

        input_frame = pd.DataFrame(
            rows,
            columns=self.feature_names,
        )

        predicted_classes = self.model.predict(input_frame)
        probabilities = self.model.predict_proba(input_frame)
        confidences = probabilities.max(axis=1)

        return [
            (int(predicted_class), float(confidence))
            for predicted_class, confidence in zip(
                predicted_classes,
                confidences,
            )
        ]