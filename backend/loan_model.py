"""Load and use the trained loan pipeline."""

from pathlib import Path
from typing import Any

import numpy as np

MODEL_PATH = Path(__file__).resolve().parent / "models" / "loan_pipeline.pkl"


def load_pipeline() -> Any:
    import joblib

    target_path = MODEL_PATH
    if not target_path.exists():
        fallback = MODEL_PATH.with_suffix(".pkl.jpg")
        if fallback.exists():
            target_path = fallback
        else:
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            try:
                import sys
                train_dir = str(Path(__file__).resolve().parent / "training")
                if train_dir not in sys.path:
                    sys.path.insert(0, train_dir)
                from backend.training.train_loan import main as train_loan
                train_loan()
            except Exception as e:
                raise FileNotFoundError(f"Loan model not found at {MODEL_PATH} and auto-training failed: {e}")
    return joblib.load(target_path)


def predict_loan(pipeline: Any, payload: dict[str, Any]) -> dict[str, Any]:
    import pandas as pd

    row = pd.DataFrame([payload])
    probability = float(pipeline.predict_proba(row)[0, 1])
    eligible = bool(pipeline.predict(row)[0])

    preprocessor = pipeline.named_steps["preprocessor"]
    estimator = pipeline.named_steps["model"]
    names = preprocessor.get_feature_names_out()
    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
    else:
        importances = np.abs(estimator.coef_[0])
    top_indices = np.argsort(importances)[::-1][:4]
    key_features = []
    observed = {}
    for i in top_indices:
        raw_name = str(names[i]).replace("num__", "").replace("cat__", "")
        if raw_name.startswith("employment_type_"):
            category = raw_name.removeprefix("employment_type_")
            label = f"employment_type ({category} category)"
            observed[label] = payload["employment_type"]
        else:
            label = raw_name
            if label in payload:
                observed[label] = payload[label]
        key_features.append({"feature": label, "importance": round(float(importances[i]), 4)})

    return {
        "prediction": "Eligible" if eligible else "Not Eligible",
        "predicted_probability": round(probability, 4),
        "key_influencing_features": key_features,
        "observed_key_feature_values": observed,
        "explanation": (
            "This experimental model's estimate is based on synthetic training data. The listed "
            "features are globally influential in the fitted model; they are not a causal or bank "
            "underwriting explanation. This result is not an actual bank approval."
        ),
        "experimental": True,
    }
