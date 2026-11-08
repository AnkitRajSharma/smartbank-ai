"""Customer-owned synthetic account, transaction, credit and report APIs."""
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
import csv
import io
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

try:
    from .auth import get_current_user
    from .json_store import JsonTable
except ImportError:
    from auth import get_current_user
    from json_store import JsonTable

DATA = Path(__file__).resolve().parent / "data"
ACCOUNTS = JsonTable(DATA / "accounts.json")
TRANSACTIONS = JsonTable(DATA / "transactions.json")
CREDIT = JsonTable(DATA / "credit_profiles.json")
ALERTS = JsonTable(DATA / "fraud_alerts.json")
REQUESTS = JsonTable(DATA / "service_requests.json")
router = APIRouter()

def _owned(table: JsonTable, user: dict[str, str], what: str) -> dict[str, Any]:
    row = next((item for item in table.read() if item.get("user_id") == user["user_id"]), None)
    if row is None:
        raise HTTPException(404, f"No {what} is available for this account")
    return row

def _transactions(user: dict[str, str]) -> list[dict[str, Any]]:
    account = _owned(ACCOUNTS, user, "account")
    return [t for t in TRANSACTIONS.read() if t.get("account_id") == account["account_id"]]

@router.get("/api/account/summary")
def account_summary(user: dict[str, str] = Depends(get_current_user)):
    account = _owned(ACCOUNTS, user, "account")
    tx = _transactions(user)
    recent = sorted(tx, key=lambda x: (x["date"], x["time"]), reverse=True)[:5]
    credit = _owned(CREDIT, user, "credit profile")
    return {"account": account, "credit_score": {"score": credit["score"], "category": credit["category"]}, "monthly_income": account["monthly_income"],
            "monthly_expenses": account["monthly_expenses"], "recent_transactions": recent}

@router.get("/api/account/details")
def account_details(user: dict[str, str] = Depends(get_current_user)):
    account = _owned(ACCOUNTS, user, "account")
    return {"account": account, "credit_profile": _owned(CREDIT, user, "credit profile")}

@router.get("/api/account/balance")
def account_balance(user: dict[str, str] = Depends(get_current_user)):
    account = _owned(ACCOUNTS, user, "account")
    return {"available_balance": account["available_balance"], "currency": "INR", "account_number": account["masked_account_number"]}

@router.get("/api/credit/profile")
def credit_profile(user: dict[str, str] = Depends(get_current_user)):
    return {"profile": _owned(CREDIT, user, "credit profile")}

@router.get("/api/credit/score")
def credit_score(user: dict[str, str] = Depends(get_current_user)):
    profile = _owned(CREDIT, user, "credit profile")
    return {"score": profile["score"], "category": profile["category"], "source": "Synthetic demonstration estimate; not a bureau score"}

@router.post("/api/credit/analyze")
def credit_analyze(user: dict[str, str] = Depends(get_current_user)):
    profile = _owned(CREDIT, user, "credit profile")
    return {"score": profile["score"], "category": profile["category"], "insights": profile["insights"],
            "disclaimer": "Synthetic demonstration profile; this is not a credit bureau record or lending decision."}

@router.get("/api/transactions")
def transactions(search: str = "", category: str = "", kind: str = Query(default="", alias="type"), status: str = "",
                 start_date: date | None = None, end_date: date | None = None,
                 sort: str = "date_desc", user: dict[str, str] = Depends(get_current_user)):
    rows = _transactions(user)
    q = search.casefold()
    rows = [r for r in rows if (not q or q in r["description"].casefold()) and
            (not category or r["merchant_category"].casefold() == category.casefold()) and
            (not kind or r["transaction_type"].casefold() == kind.casefold()) and
            (not status or r["status"].casefold() == status.casefold()) and
            (not start_date or r["date"] >= str(start_date)) and (not end_date or r["date"] <= str(end_date))]
    rows.sort(key=lambda r: (float(r["amount"]), r["date"]) if sort == "amount_asc" else
              (float(r["amount"]), r["date"]) if sort == "amount_desc" else (r["date"], r["time"]), reverse=sort in {"amount_desc", "date_desc"})
    return {"transactions": rows, "total": len(rows)}

@router.get("/api/transactions/recent")
def recent_transactions(limit: int = Query(10, ge=1, le=50), user: dict[str, str] = Depends(get_current_user)):
    rows = sorted(_transactions(user), key=lambda r: (r["date"], r["time"]), reverse=True)
    return {"transactions": rows[:limit]}

@router.get("/api/transactions/analytics")
def transaction_analytics(user: dict[str, str] = Depends(get_current_user)):
    rows = _transactions(user)
    income = sum(float(r["amount"]) for r in rows if r["transaction_type"] == "Credit")
    debits = [r for r in rows if r["transaction_type"] == "Debit"]
    expenses = sum(float(r["amount"]) for r in debits)
    categories = Counter()
    monthly = defaultdict(lambda: {"income": 0.0, "expenses": 0.0})
    for row in rows:
        amount = float(row["amount"])
        month = row["date"][:7]
        if row["transaction_type"] == "Credit": monthly[month]["income"] += amount
        else:
            monthly[month]["expenses"] += amount
            categories[row["merchant_category"]] += amount
    values = [float(r["amount"]) for r in rows]
    return {"total_income": round(income, 2), "total_expenditure": round(expenses, 2), "net_savings": round(income-expenses, 2),
            "average_transaction_amount": round(sum(values)/len(values), 2) if values else 0,
            "highest_transaction": max(values, default=0), "lowest_transaction": min(values, default=0),
            "category_expenses": [{"category": k, "amount": round(v, 2)} for k,v in sorted(categories.items())],
            "monthly_trends": [{"month": k, **{a: round(b,2) for a,b in v.items()}} for k,v in sorted(monthly.items())]}

@router.get("/api/transactions/{transaction_id}")
def transaction_detail(transaction_id: str, user: dict[str, str] = Depends(get_current_user)):
    row = next((item for item in _transactions(user) if item.get("transaction_id") == transaction_id), None)
    if row is None:
        raise HTTPException(404, "Transaction not found")
    return {"transaction": row}

@router.get("/api/fraud/alerts")
def fraud_alerts(user: dict[str, str] = Depends(get_current_user)):
    account = _owned(ACCOUNTS, user, "account")
    return {"alerts": [a for a in ALERTS.read() if a.get("account_id") == account["account_id"]]}

@router.get("/api/agent/tools")
def agent_tools(user: dict[str, str] = Depends(get_current_user)):
    return {"tools": ["get_account_balance", "get_recent_transactions", "get_spending_summary", "get_credit_profile", "banking_faq"], "mode": "local"}

@router.get("/api/agent/conversations")
def agent_conversations(user: dict[str, str] = Depends(get_current_user)):
    table = JsonTable(DATA / "conversations.json")
    rows = [row for row in table.read() if row.get("user_id") == user["user_id"]]
    rows.sort(key=lambda row: row.get("timestamp", ""), reverse=True)
    return {"conversations": rows[:100]}

@router.post("/api/rag/query")
def rag_query(payload: dict[str, str], user: dict[str, str] = Depends(get_current_user)):
    try:
        from .chatbot import gemini_answer, get_bot_response
        from .rag_service import retrieve
    except ImportError:
        from chatbot import gemini_answer, get_bot_response
        from rag_service import retrieve
    question = payload.get("query", "").strip()
    if not question: raise HTTPException(422, "query is required")
    docs = retrieve(question)
    context = "\n".join(f"[{d['title']}] {d['text']}" for d in docs)
    answer = gemini_answer(question, context) if docs else None
    return {"answer": answer or get_bot_response(question), "sources": [{"title": d["title"], "reference": d["source"], "relevance": d["score"]} for d in docs], "mode": "gemini_rag" if answer else "local_retrieval_faq"}

@router.get("/api/rag/sources")
def rag_sources(user: dict[str, str] = Depends(get_current_user)):
    try:
        from .rag_service import ROOT
    except ImportError:
        from rag_service import ROOT
    return {"sources": [{"title": p.stem.replace("_", " ").title(), "reference": p.name} for p in sorted(ROOT.glob("*.md"))]}

@router.post("/api/reports/generate")
def generate_report(user: dict[str, str] = Depends(get_current_user)):
    analytics = transaction_analytics(user)
    account = _owned(ACCOUNTS, user, "account")
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    report = {"report_id": str(uuid.uuid4()), "user_id": user["user_id"], "period": datetime.now().strftime("%Y-%m"),
              "generated_at": now, "account": account["masked_account_number"], "analytics": analytics}
    REQUESTS.append(report)
    return {"report": report}

@router.get("/api/reports/monthly")
def monthly_report(month: str | None = None, user: dict[str, str] = Depends(get_current_user)):
    period = month or datetime.now().strftime("%Y-%m")
    if len(period) != 7 or period[4] != "-" or not period[:4].isdigit() or not period[5:].isdigit() or not 1 <= int(period[5:]) <= 12:
        raise HTTPException(422, "month must use YYYY-MM format")
    rows = [row for row in _transactions(user) if row["date"].startswith(period)]
    income = sum(float(row["amount"]) for row in rows if row["transaction_type"] == "Credit")
    expenses = sum(float(row["amount"]) for row in rows if row["transaction_type"] == "Debit")
    categories: dict[str, float] = defaultdict(float)
    for row in rows:
        if row["transaction_type"] == "Debit":
            categories[row["merchant_category"]] += float(row["amount"])
    report = {"transaction_count": len(rows), "total_income": round(income, 2),
              "total_expenditure": round(expenses, 2), "net_savings": round(income-expenses, 2),
              "category_expenses": [{"category": key, "amount": round(value, 2)} for key, value in sorted(categories.items())]}
    report["month"] = period
    report["disclaimer"] = "Synthetic transaction report for demonstration."
    return {"report": report}
