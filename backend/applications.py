"""Customer-owned loan application submission and history endpoints."""

import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status

try:
    from .auth import get_current_user
    from .json_store import JsonTable
    from .loan_model import predict_loan
    from .schemas import LoanApplicationSubmission
except ImportError:
    from auth import get_current_user
    from json_store import JsonTable
    from loan_model import predict_loan
    from schemas import LoanApplicationSubmission

DATA_DIR = Path(__file__).resolve().parent / "data"
APPLICATIONS_TABLE = JsonTable(DATA_DIR / "loan_applications.json")
LOW_RISK_MIN_PROBABILITY = float(os.getenv("SMARTBANK_LOW_RISK_MIN_PROBABILITY", "0.75"))
MEDIUM_RISK_MIN_PROBABILITY = float(os.getenv("SMARTBANK_MEDIUM_RISK_MIN_PROBABILITY", "0.45"))
if not 0 <= MEDIUM_RISK_MIN_PROBABILITY < LOW_RISK_MIN_PROBABILITY <= 1:
    raise RuntimeError("Risk thresholds must satisfy 0 <= medium < low <= 1")

router = APIRouter(prefix="/api/loan", tags=["loan applications"])


def risk_category(probability: float) -> tuple[float, str]:
    risk_score = round((1.0 - probability) * 100, 1)
    if probability >= LOW_RISK_MIN_PROBABILITY:
        category = "Low"
    elif probability >= MEDIUM_RISK_MIN_PROBABILITY:
        category = "Medium"
    else:
        category = "High"
    return risk_score, category


def _to_public_application(row: dict[str, Any]) -> dict[str, Any]:
    return {
        **row,
        "age": int(row["age"]),
        "annual_income": float(row["annual_income"]),
        "employment_experience": float(row["employment_experience"]),
        "credit_score": int(row["credit_score"]),
        "existing_loan_amount": float(row["existing_loan_amount"]),
        "monthly_debt": float(row["monthly_debt"]),
        "requested_loan_amount": float(row["requested_loan_amount"]),
        "loan_probability": float(row["loan_probability"]),
        "risk_score": float(row["risk_score"]),
    }


@router.post("/apply", status_code=status.HTTP_201_CREATED)
def apply_for_loan(
    payload: LoanApplicationSubmission,
    request: Request,
    user: dict[str, str] = Depends(get_current_user),
) -> dict[str, Any]:
    if user.get("role") != "customer":
        raise HTTPException(status_code=403, detail="Only customer accounts can submit loan applications")
    pipeline = getattr(request.app.state, "loan_pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Loan prediction model is unavailable")
    application_values = payload.model_dump()
    applicant_name = application_values.pop("applicant_name")
    prediction_result = predict_loan(pipeline, application_values)
    probability = float(prediction_result["predicted_probability"])
    risk_score, risk = risk_category(probability)
    application_id = f"SB-{uuid.uuid4().hex[:12].upper()}"
    submitted_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    record = {
        "application_id": application_id,
        "user_id": user["user_id"],
        "applicant_name": applicant_name,
        **application_values,
        "loan_probability": probability,
        "risk_score": risk_score,
        "risk_category": risk,
        "prediction": prediction_result["prediction"],
        "application_status": "Under Review",
        "application_date": submitted_at,
    }
    try:
        APPLICATIONS_TABLE.append(record)
    except RuntimeError as error:
        raise HTTPException(status_code=500, detail="Could not save the loan application") from error
    return {
        "application_id": application_id,
        "applicant_name": applicant_name,
        "requested_loan_amount": application_values["requested_loan_amount"],
        "prediction": prediction_result["prediction"],
        "loan_probability": probability,
        "risk_score": risk_score,
        "risk_category": risk,
        "application_status": "Under Review",
        "application_date": submitted_at,
        "message": "Application saved for review. This experimental model is not a lending decision.",
    }


@router.get("/my-applications")
def my_applications(user: dict[str, str] = Depends(get_current_user)) -> dict[str, Any]:
    try:
        rows = [row for row in APPLICATIONS_TABLE.read() if row.get("user_id") == user["user_id"]]
    except RuntimeError as error:
        raise HTTPException(status_code=500, detail="Could not read application history") from error
    rows.sort(key=lambda row: row.get("application_date", ""), reverse=True)
    return {"applications": [_to_public_application(row) for row in rows]}


@router.get("/applications/{application_id}")
def get_application(
    application_id: str,
    user: dict[str, str] = Depends(get_current_user),
) -> dict[str, Any]:
    try:
        row = next((item for item in APPLICATIONS_TABLE.read() if item.get("application_id") == application_id), None)
    except RuntimeError as error:
        raise HTTPException(status_code=500, detail="Could not read the application") from error
    if row is None or row.get("user_id") != user["user_id"]:
        raise HTTPException(status_code=404, detail="Application not found")
    return {"application": _to_public_application(row)}
