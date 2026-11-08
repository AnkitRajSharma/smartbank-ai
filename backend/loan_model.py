"""Load and use the trained loan pipeline."""

from pathlib import Path
from typing import Any

import numpy as np

MODEL_PATH = Path(__file__).resolve().parent / "models" / "loan_pipeline.pkl"


def load_pipeline() -> Any:
    import joblib

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Loan model not found at {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


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
