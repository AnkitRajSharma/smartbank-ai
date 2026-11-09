"""Password hashing and short-lived JWT session helpers."""

import os
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

import jwt
from pwdlib import PasswordHash

PASSWORD_HASHER = PasswordHash.recommended()
JWT_ALGORITHM = "HS256"
JWT_SECRET = os.getenv("SMARTBANK_JWT_SECRET", "local-demo-only-change-this-secret-before-deployment-2026")
JWT_TTL_SECONDS = int(os.getenv("SMARTBANK_JWT_TTL_SECONDS", "7200"))
COOKIE_NAME = "smartbank_session"
COOKIE_SECURE = os.getenv("SMARTBANK_COOKIE_SECURE", "false").lower() == "true"
_active_sessions: set[str] = set()
_session_lock = threading.Lock()


def hash_password(password: str) -> str:
    return PASSWORD_HASHER.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return PASSWORD_HASHER.verify(password, password_hash)
    except Exception:
        return False


def create_access_token(user_id: str) -> tuple[str, str]:
    now = datetime.now(timezone.utc)
    jti = str(uuid.uuid4())
    token = jwt.encode(
        {"sub": user_id, "jti": jti, "iat": now, "exp": now + timedelta(seconds=JWT_TTL_SECONDS)},
        JWT_SECRET,
        algorithm=JWT_ALGORITHM,
    )
    with _session_lock:
        _active_sessions.add(jti)
    return token, jti


def decode_access_token(token: str) -> dict[str, Any] | None:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.PyJWTError:
        return None
    jti = payload.get("jti")
    with _session_lock:
        if not isinstance(jti, str) or jti not in _active_sessions:
            return None
    return payload


def revoke_access_token(token: str | None) -> None:
    if not token:
        return
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM], options={"verify_exp": False})
    except jwt.PyJWTError:
        return
    jti = payload.get("jti")
    if isinstance(jti, str):
        with _session_lock:
            _active_sessions.discard(jti)
