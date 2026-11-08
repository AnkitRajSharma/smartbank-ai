"""FastAPI application for SmartBank AI chat, authentication, predictions, and analytics."""

from contextlib import asynccontextmanager
import logging
from typing import Any
from pathlib import Path

# Uvicorn does not load .env files unless configured explicitly. Load this project's
# local settings before importing auth/security modules that read environment values.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError, field_validator

try:
    from .applications import router as applications_router
    from .banking import router as banking_router
    from .auth import get_current_user, router as auth_router
    from .chatbot import ChatbotService
    from .fraud_model import load_pipeline as load_fraud_pipeline, predict_fraud
    from .loan_model import load_pipeline as load_loan_pipeline, predict_loan
    from .portfolio import router as portfolio_router
    from .schemas import FraudTransaction, LoanApplication
    from .json_store import JsonTable
except ImportError:
    from applications import router as applications_router
    from banking import router as banking_router
    from auth import get_current_user, router as auth_router
    from chatbot import ChatbotService
    from fraud_model import load_pipeline as load_fraud_pipeline, predict_fraud
    from loan_model import load_pipeline as load_loan_pipeline, predict_loan
    from portfolio import router as portfolio_router
    from schemas import FraudTransaction, LoanApplication
    from json_store import JsonTable

chatbot = ChatbotService()
logger = logging.getLogger(__name__)
CONVERSATIONS = JsonTable(Path(__file__).resolve().parent / "data" / "conversations.json")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load each model once on startup; missing artifacts disable only their own endpoints."""
    try:
        app.state.loan_pipeline = load_loan_pipeline()
    except Exception:
        logger.warning("Loan prediction pipeline is unavailable; train or repair its model artifact.")
        app.state.loan_pipeline = None
    try:
        app.state.fraud_pipeline = load_fraud_pipeline()
    except Exception:
        logger.warning("Fraud prediction pipeline is unavailable; train or repair its model artifact.")
        app.state.fraud_pipeline = None
    yield


app = FastAPI(title="SmartBank AI API", version="4.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500", "http://localhost:5500",
        "http://127.0.0.1:5501", "http://localhost:5501",
        "http://127.0.0.1:5502", "http://localhost:5502",
        "http://127.0.0.1:8000", "http://localhost:8000",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type"],
)
app.include_router(auth_router)
app.include_router(applications_router)
app.include_router(portfolio_router)
app.include_router(banking_router)
app.mount("/app", StaticFiles(directory=Path(__file__).resolve().parent.parent / "frontend", html=True), name="frontend")


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    session_id: str | None = Field(default=None, min_length=1, max_length=100)

    @field_validator("message")
    @classmethod
    def message_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Message cannot be blank")
        return value

    @field_validator("session_id")
    @classmethod
    def session_id_must_not_be_blank(cls, value: str | None) -> str | None:
        if value is not None:
            value = value.strip()
            if not value:
                raise ValueError("Session ID cannot be blank")
        return value


class ChatResponse(BaseModel):
    user_message: str
    bot_response: str
    status: str


def _require_model(model: Any | None, model_name: str) -> Any:
    if model is None:
        raise HTTPException(status_code=503, detail=f"{model_name} model is unavailable. Train the model and restart the backend.")
    return model


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "SmartBank AI Backend is running"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.get("/api/models/status")
def models_status(user: dict[str, str] = Depends(get_current_user)) -> dict[str, Any]:
    loan_loaded = getattr(app.state, "loan_pipeline", None) is not None
    fraud_loaded = getattr(app.state, "fraud_pipeline", None) is not None
    return {
        "loan_model_loaded": loan_loaded,
        "fraud_model_loaded": fraud_loaded,
        "status": "ready" if loan_loaded and fraud_loaded else "partially_available" if loan_loaded or fraud_loaded else "unavailable",
        "predictions_are_experimental": True,
    }


@app.post("/api/loan/predict")
def loan_prediction(request: LoanApplication, user: dict[str, str] = Depends(get_current_user)) -> dict[str, Any]:
    return predict_loan(_require_model(getattr(app.state, "loan_pipeline", None), "Loan"), request.model_dump())


@app.post("/api/fraud/predict")
def fraud_prediction(request: FraudTransaction, user: dict[str, str] = Depends(get_current_user)) -> dict[str, Any]:
    return predict_fraud(_require_model(getattr(app.state, "fraud_pipeline", None), "Fraud"), request.model_dump())


@app.post("/api/agent/chat", response_model=ChatResponse)
@app.post("/api/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    user: dict[str, str] = Depends(get_current_user),
) -> ChatResponse:
    loan_model = getattr(app.state, "loan_pipeline", None)
    fraud_model = getattr(app.state, "fraud_pipeline", None)

    def validated_loan(values: dict[str, Any]) -> dict[str, Any]:
        try:
            clean = LoanApplication.model_validate(values).model_dump()
        except ValidationError as error:
            raise ValueError("Please check the loan details against the application form limits") from error
        return predict_loan(_require_model(loan_model, "Loan"), clean)

    def validated_fraud(values: dict[str, Any]) -> dict[str, Any]:
        try:
            clean = FraudTransaction.model_validate(values).model_dump()
        except ValidationError as error:
            raise ValueError("Please check the transaction details and allowed category values") from error
        return predict_fraud(_require_model(fraud_model, "Fraud"), clean)

    normalized = request.message.lower()
    # Safe, read-only tools are allowlisted and scoped to the authenticated customer's records.
    if any(term in normalized for term in ("my balance", "account balance", "how much do i have")):
        try: from .banking import ACCOUNTS
        except ImportError: from banking import ACCOUNTS
        account = next((row for row in ACCOUNTS.read() if row.get("user_id") == user["user_id"]), None)
        response = (f"Your available synthetic demo balance is ₹{account['available_balance']:,.0f} "
                    f"(account {account['masked_account_number']})." if account else "No demo account is linked to this profile.")
    elif any(term in normalized for term in ("my spending", "my expenses", "spending summary")):
        try: from .banking import transaction_analytics
        except ImportError: from banking import transaction_analytics
        summary = transaction_analytics(user)
        response = f"Across the synthetic transaction history, income is ₹{summary['total_income']:,.0f}, expenditure is ₹{summary['total_expenditure']:,.0f}, and net savings are ₹{summary['net_savings']:,.0f}."
    elif any(term in normalized for term in ("my credit score", "my credit profile", "my credit")):
        try: from .banking import CREDIT
        except ImportError: from banking import CREDIT
        profile = next((item for item in CREDIT.read() if item.get("user_id") == user["user_id"]), None)
        response = (f"Your illustrative synthetic demo score is {profile['score']} ({profile['category']}). "
                    f"Utilization is {profile['utilization_percent']}% and the demo debt-to-income ratio is {profile['debt_to_income_ratio']:.1%}. "
                    "This is not a credit bureau score or lending decision." if profile else "No demo credit profile is linked to this account.")
    else:
        response = chatbot.respond(
        request.message,
        f"{user['user_id']}:{request.session_id or 'default'}",
        validated_loan if loan_model is not None else None,
        validated_fraud if fraud_model is not None else None,
        )
    try:
        CONVERSATIONS.append({"user_id": user["user_id"], "session_id": request.session_id or "default",
                              "timestamp": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(timespec="seconds"),
                              "user_message": request.message, "assistant_message": response})
    except RuntimeError:
        logger.warning("Conversation history could not be saved")
    return ChatResponse(user_message=request.message, bot_response=response, status="success")


@app.delete("/api/chat/session/{session_id}")
def clear_chat_session(
    session_id: str,
    user: dict[str, str] = Depends(get_current_user),
) -> dict[str, str]:
    # Prefix sessions by user so a customer cannot clear another customer's chat state.
    chatbot.sessions.pop(f"{user['user_id']}:{session_id}", None)
    return {"status": "cleared"}
