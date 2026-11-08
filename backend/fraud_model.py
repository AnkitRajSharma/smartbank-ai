"""Load and use the trained fraud risk pipeline."""

from pathlib import Path
from typing import Any

MODEL_PATH = Path(__file__).resolve().parent / "models" / "fraud_pipeline.pkl"


def load_pipeline() -> Any:
    import joblib

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Fraud model not found at {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


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
