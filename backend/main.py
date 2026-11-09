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

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
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
