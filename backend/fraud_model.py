"""Load and use the trained fraud risk pipeline."""

from pathlib import Path
from typing import Any

MODEL_PATH = Path(__file__).resolve().parent / "models" / "fraud_pipeline.pkl"


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
                from backend.training.train_fraud import main as train_fraud
                train_fraud()
            except Exception as e:
                raise FileNotFoundError(f"Fraud model not found at {MODEL_PATH} and auto-training failed: {e}")
    return joblib.load(target_path)


def predict_fraud(pipeline: Any, payload: dict[str, Any]) -> dict[str, Any]:
    import pandas as pd

    probability = float(pipeline.predict_proba(pd.DataFrame([payload]))[0, 1])
    prediction = "Suspicious Transaction" if bool(pipeline.predict(pd.DataFrame([payload]))[0]) else "Normal"
    risk_level = "High" if probability >= 0.60 else "Medium" if probability >= 0.20 else "Low"

    characteristics = []
    if payload["international_transaction"]:
        characteristics.append("international transaction")
    if payload["transaction_hour"] < 5 or payload["transaction_hour"] >= 23:
        characteristics.append("unusual hour")
    if payload["distance_from_usual_location_km"] >= 500:
        characteristics.append("large distance from usual location")
    if payload["transactions_last_24_hours"] >= 8:
        characteristics.append("high recent transaction count")
    if payload["transaction_amount"] >= 100_000:
        characteristics.append("large transaction amount")
    if not characteristics:
        characteristics.append("no individual high-risk characteristic from the simple review rules")

    recommendation = (
        "Verify the transaction through official banking channels. This prediction does not block a transaction."
        if risk_level == "High"
        else "If you do not recognize the transaction, contact your bank using its official channels."
    )
    return {
        "prediction": prediction,
        "fraud_probability": round(probability, 4),
        "risk_level": risk_level,
        "main_suspicious_characteristics": characteristics,
        "recommendation": recommendation,
        "experimental": True,
    }
