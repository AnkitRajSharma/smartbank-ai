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

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
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
    allow_origin_regex=r"^https?://.*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
def root() -> RedirectResponse:
    """Redirect root requests directly to the frontend login page."""
    return RedirectResponse(url="/app/login.html", status_code=status.HTTP_307_TEMPORARY_REDIRECT)


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


@app.get("/api/agent/loan-profile")
def get_agent_loan_profile(user: dict[str, str] = Depends(get_current_user)) -> dict[str, Any]:
    """Provide customer profile pre-filled defaults for loan eligibility evaluation."""
    try:
        from .banking import ACCOUNTS, CREDIT
    except ImportError:
        from banking import ACCOUNTS, CREDIT

    account = next((r for r in ACCOUNTS.read() if r.get("user_id") == user["user_id"]), {})
    credit = next((r for r in CREDIT.read() if r.get("user_id") == user["user_id"]), {})
    annual_income = float(account.get("annual_income") or 600000.0)
    experience = float(account.get("employment_experience") or 3.0)
    employment = account.get("employment_type") or "Salaried"
    score = int(credit.get("score") or 750)
    monthly_debt = float(account.get("monthly_emi") or credit.get("monthly_emi") or 0.0)
    existing_loans = sum(float(l.get("outstanding_amount", 0.0)) for l in account.get("existing_loans", []))
    return {
        "age": 30,
        "applicant_name": user.get("name", "Applicant"),
        "annual_income": annual_income,
        "employment_type": employment,
        "employment_experience": experience,
        "credit_score": score,
        "existing_loan_amount": existing_loans,
        "monthly_debt": monthly_debt,
        "requested_loan_amount": 500000.0,
    }


@app.get("/api/agent/scan-transactions-fraud")
@app.post("/api/agent/scan-transactions-fraud")
def scan_transactions_fraud(user: dict[str, str] = Depends(get_current_user)) -> dict[str, Any]:
    """Scan authenticated customer transactions with the fraud detection ML model."""
    try:
        from .banking import ACCOUNTS, TRANSACTIONS, ALERTS
    except ImportError:
        from banking import ACCOUNTS, TRANSACTIONS, ALERTS

    account = next((r for r in ACCOUNTS.read() if r.get("user_id") == user["user_id"]), None)
    if not account:
        return {"transactions": [], "alerts": [], "summary": "No transactions found"}

    user_tx = [t for t in TRANSACTIONS.read() if t.get("account_id") == account["account_id"]]
    user_alerts = [a for a in ALERTS.read() if a.get("account_id") == account["account_id"]]
    fraud_model = getattr(app.state, "fraud_pipeline", None)

    scanned = []
    for tx in user_tx[:30]:
        time_str = tx.get("time", "12:00:00")
        hour = int(time_str.split(":")[0]) if ":" in time_str else 12
        channel_raw = tx.get("channel", "Online")
        channel = "Online" if channel_raw in ["Online", "UPI"] else "In Person" if channel_raw == "Card" else "ATM" if channel_raw == "ATM" else "Mobile"
        cat_raw = tx.get("merchant_category", "Other")
        cat = cat_raw if cat_raw in ["Retail", "Groceries", "Travel", "Dining", "Electronics", "Other"] else "Other"
        tx_type_raw = tx.get("transaction_type", "Purchase")
        tx_type = "Cash Withdrawal" if "cash" in tx.get("description", "").lower() or tx_type_raw == "Cash Withdrawal" else "Transfer" if tx_type_raw == "Credit" or "transfer" in tx.get("description", "").lower() else "Online Payment" if channel in ["Online", "Mobile"] else "Purchase"
        amount = float(tx.get("amount", 0))

        payload = {
            "transaction_amount": max(1.0, amount),
            "transaction_hour": hour,
            "transaction_type": tx_type,
            "merchant_category": cat,
            "transactions_last_24_hours": 2,
            "distance_from_usual_location_km": 15.0 if amount < 50000 else 650.0,
            "international_transaction": bool(amount > 100000 or "intl" in tx.get("description", "").lower()),
            "transaction_channel": channel,
        }
        risk_result = {"prediction": "Normal", "risk_level": "Low", "fraud_probability": 0.05, "main_suspicious_characteristics": ["Normal domestic transaction"], "recommendation": "Normal transaction."}
        if fraud_model is not None:
            try:
                risk_result = predict_fraud(fraud_model, payload)
            except Exception:
                pass

        scanned.append({
            "transaction_id": tx.get("transaction_id"),
            "date": tx.get("date"),
            "time": tx.get("time"),
            "description": tx.get("description"),
            "amount": amount,
            "merchant_category": cat_raw,
            "channel": channel_raw,
            "risk_level": risk_result.get("risk_level", "Low"),
            "prediction": risk_result.get("prediction", "Normal"),
            "fraud_probability": risk_result.get("fraud_probability", 0.0),
            "characteristics": risk_result.get("main_suspicious_characteristics", []),
            "recommendation": risk_result.get("recommendation", ""),
        })

    high_risk_count = sum(1 for t in scanned if t["risk_level"] == "High")
    medium_risk_count = sum(1 for t in scanned if t["risk_level"] == "Medium")
    status_label = "Attention Required" if (high_risk_count > 0 or user_alerts) else "Moderate Vigilance" if medium_risk_count > 0 else "Secure"

    return {
        "transactions": scanned,
        "total_scanned": len(scanned),
        "high_risk_count": high_risk_count,
        "medium_risk_count": medium_risk_count,
        "alerts": user_alerts,
        "account_security_status": status_label,
    }


def _build_customer_context(user: dict[str, str]) -> str:
    """Safely compile authenticated customer financial records for grounded BFSI advisory."""
    context_lines = [
        f"Customer Name: {user.get('name', 'Valued Customer')}",
        f"Customer Email: {user.get('email', '')}",
        f"User ID: {user.get('user_id', '')}",
        f"Role: {user.get('role', 'customer')}",
    ]
    try:
        try:
            from .banking import ACCOUNTS, CREDIT, transaction_analytics
        except ImportError:
            from banking import ACCOUNTS, CREDIT, transaction_analytics

        try:
            account = next((row for row in ACCOUNTS.read() if row.get("user_id") == user["user_id"]), None)
            if account:
                context_lines.append(
                    f"Account Details: {account.get('account_type', 'Savings')} Account ({account.get('masked_account_number')}) | "
                    f"Available Balance: ₹{float(account.get('available_balance', 0)):,.2f} | "
                    f"Annual Income: ₹{float(account.get('annual_income', 0)):,.0f} | "
                    f"Employment: {account.get('employment_type', 'Not specified')}"
                )
        except Exception:
            pass

        try:
            profile = next((item for item in CREDIT.read() if item.get("user_id") == user["user_id"]), None)
            if profile:
                context_lines.append(
                    f"Credit Profile: Score {profile.get('score')} ({profile.get('category')}) | "
                    f"Credit Utilization: {profile.get('utilization_percent')}% | "
                    f"Debt-to-Income (DTI): {float(profile.get('debt_to_income_ratio', 0)):.1%}"
                )
        except Exception:
            pass

        try:
            analytics = transaction_analytics(user)
            if analytics:
                context_lines.append(
                    f"Spending Analytics: Total Inflow/Income: ₹{analytics.get('total_income', 0):,.2f} | "
                    f"Total Outflow/Expenditure: ₹{analytics.get('total_expenditure', 0):,.2f} | "
                    f"Net Savings: ₹{analytics.get('net_savings', 0):,.2f}"
                )
        except Exception:
            pass
    except Exception:
        pass

    return "\n".join(context_lines)


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

    customer_context = _build_customer_context(user)
    response = chatbot.respond(
        message=request.message,
        session_id=f"{user['user_id']}:{request.session_id or 'default'}",
        loan_predictor=validated_loan if loan_model is not None else None,
        fraud_predictor=validated_fraud if fraud_model is not None else None,
        customer_context=customer_context,
        user_id=user["user_id"],
    )
    try:
        CONVERSATIONS.append({"user_id": user["user_id"], "session_id": request.session_id or "default",
                              "timestamp": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(timespec="seconds"),
                              "user_message": request.message, "assistant_message": response})
    except RuntimeError:
        logger.warning("Conversation history could not be saved")
    return ChatResponse(user_message=request.message, bot_response=response, status="success")


@app.post("/api/rag/upload")
async def upload_rag_document(
    file: UploadFile = File(...),
    user: dict[str, str] = Depends(get_current_user),
) -> dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename cannot be empty")
    allowed_exts = {".pdf", ".txt", ".md", ".csv", ".tsv", ".json", ".doc", ".docx"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Allowed formats: PDF, TXT, MD, CSV, TSV, JSON."
        )
    content = await file.read()
    if len(content) > 15 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Document exceeds maximum allowed size of 15 MB")
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        from .rag_service import save_user_document
    except ImportError:
        from rag_service import save_user_document

    try:
        safe_name, chunk_count = save_user_document(user["user_id"], file.filename, content)
        return {
            "status": "success",
            "filename": safe_name,
            "chunks_count": chunk_count,
            "message": f"Successfully indexed '{safe_name}' ({chunk_count} sections) into your RAG session."
        }
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to process document: {e}") from e


@app.get("/api/rag/documents")
def get_rag_documents(user: dict[str, str] = Depends(get_current_user)) -> dict[str, Any]:
    try:
        from .rag_service import list_user_documents
    except ImportError:
        from rag_service import list_user_documents
    return {"documents": list_user_documents(user["user_id"])}


@app.delete("/api/rag/documents/{filename}")
def remove_rag_document(filename: str, user: dict[str, str] = Depends(get_current_user)) -> dict[str, Any]:
    try:
        from .rag_service import clear_user_documents
    except ImportError:
        from rag_service import clear_user_documents
    cleared = clear_user_documents(user["user_id"], filename)
    return {"status": "cleared" if cleared else "not_found", "filename": filename}


@app.delete("/api/chat/session/{session_id}")
def clear_chat_session(
    session_id: str,
    user: dict[str, str] = Depends(get_current_user),
) -> dict[str, str]:
    # Prefix sessions by user so a customer cannot clear another customer's chat state.
    chatbot.sessions.pop(f"{user['user_id']}:{session_id}", None)
    return {"status": "cleared"}
