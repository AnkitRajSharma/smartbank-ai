"""Registration, login, logout, and cookie-backed authenticated user dependencies."""

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Cookie, Depends, HTTPException, Request, Response, status

try:
    from .json_store import JsonTable
    from .schemas import LoginRequest, RegisterRequest
    from .security import (COOKIE_NAME, COOKIE_SECURE, JWT_TTL_SECONDS, create_access_token,
                      decode_access_token, hash_password, revoke_access_token,
                      verify_password)
except ImportError:
    from json_store import JsonTable
    from schemas import LoginRequest, RegisterRequest
    from security import (COOKIE_NAME, COOKIE_SECURE, JWT_TTL_SECONDS, create_access_token,
                      decode_access_token, hash_password, revoke_access_token, verify_password)

DATA_DIR = Path(__file__).resolve().parent / "data"
USERS_COLUMNS = ["user_id", "name", "email", "password_hash", "role", "created_at"]
USERS_TABLE = JsonTable(DATA_DIR / "users.json")
router = APIRouter(prefix="/api/auth", tags=["authentication"])


def public_user(user: dict[str, Any]) -> dict[str, str]:
    return {key: str(user[key]) for key in ["user_id", "name", "email", "role"]}


def find_user_by_email(email: str) -> dict[str, str] | None:
    email = email.strip().lower()
    return next((user for user in USERS_TABLE.read() if user.get("email", "").lower() == email), None)


def find_user_by_id(user_id: str) -> dict[str, str] | None:
    return next((user for user in USERS_TABLE.read() if user.get("user_id") == user_id), None)


def get_current_user(
    request: Request,
    token: str | None = Cookie(default=None, alias=COOKIE_NAME),
) -> dict[str, str]:
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    claims = decode_access_token(token)
    user = find_user_by_id(str(claims.get("sub"))) if claims else None
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Session is invalid or expired")
    request.state.session_token = token
    return user


def require_admin(user: dict[str, str] = Depends(get_current_user)) -> dict[str, str]:
    if user.get("role") != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Analytics administrator access required")
    return user


def _set_auth_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key=COOKIE_NAME,
        value=token,
        max_age=JWT_TTL_SECONDS,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite="lax",
        path="/",
    )


@router.post("/register", status_code=status.HTTP_201_CREATED)
def register(payload: RegisterRequest) -> dict[str, Any]:
    user = {
        "user_id": str(uuid.uuid4()),
        "name": payload.name,
        "email": payload.email,
        "password_hash": hash_password(payload.password),
        "role": "customer",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if not USERS_TABLE.append_unique(user, "email"):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="An account with this email already exists")
    # Give newly registered demo customers an isolated, zero-balance synthetic profile.
    try:
        try:
            from .json_store import JsonTable as Table
        except ImportError:
            from json_store import JsonTable as Table
        import secrets
        account_id = f"ACC-{uuid.uuid4().hex[:12].upper()}"
        account = {"user_id": user["user_id"], "customer_id": user["user_id"], "name": user["name"],
                   "email": user["email"], "account_id": account_id,
                   "masked_account_number": f"XXXX XXXX {secrets.randbelow(9000)+1000}",
                   "ifsc": "SMBK0000420", "account_type": "Savings", "available_balance": 0,
                   "opening_balance": 0, "opening_date": datetime.now(timezone.utc).date().isoformat(),
                   "annual_income": 0, "employment_type": "Not provided", "employment_experience": 0,
                   "monthly_income": 0, "monthly_expenses": 0, "existing_loans": [], "monthly_emi": 0,
                   "loan_history": [], "synthetic_data": True}
        profiles = {"score": 600, "category": "Illustrative", "utilization_percent": 0,
                    "monthly_emi": 0, "debt_to_income_ratio": 0, "repayment_history": "Not available",
                    "insights": ["Add financial details with a lender to assess an actual credit profile."]}
        Table(DATA_DIR / "accounts.json").append(account)
        Table(DATA_DIR / "credit_profiles.json").append({"user_id": user["user_id"], **profiles})
    except RuntimeError as error:
        raise HTTPException(status_code=500, detail="Account was created but its synthetic profile could not be initialized") from error
    return {"user": public_user(user), "message": "Registration successful. Please sign in."}


@router.post("/login")
def login(payload: LoginRequest, response: Response) -> dict[str, Any]:
    user = find_user_by_email(payload.email)
    if user is None or not verify_password(payload.password, user.get("password_hash", "")):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Email or password is incorrect")
    token, _jti = create_access_token(user["user_id"])
    _set_auth_cookie(response, token)
    return {"user": public_user(user), "message": "Signed in successfully"}


@router.post("/logout")
def logout(response: Response, token: str | None = Cookie(default=None, alias=COOKIE_NAME)) -> dict[str, str]:
    revoke_access_token(token)
    response.delete_cookie(key=COOKIE_NAME, path="/", httponly=True, secure=COOKIE_SECURE, samesite="lax")
    return {"status": "logged_out"}


@router.get("/me")
def me(user: dict[str, str] = Depends(get_current_user)) -> dict[str, Any]:
    return {"user": public_user(user)}
