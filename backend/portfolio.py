"""Role-protected aggregate-only portfolio analytics from application JSON records."""

from collections import Counter, defaultdict
from statistics import mean, median
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

try:
    from .applications import APPLICATIONS_TABLE
    from .auth import require_admin
except ImportError:
    from applications import APPLICATIONS_TABLE
    from auth import require_admin

router = APIRouter(prefix="/api/portfolio", tags=["portfolio analytics"], dependencies=[Depends(require_admin)])
RISK_ORDER = ["Low", "Medium", "High"]


def _records() -> list[dict[str, Any]]:
    try:
        return APPLICATIONS_TABLE.read()
    except RuntimeError as error:
        raise HTTPException(status_code=500, detail="Could not read portfolio records") from error


def _number(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return 0.0


def _month(row: dict[str, str]) -> str:
    return row.get("application_date", "")[:7]


def _fixed_histogram(values: list[float], boundaries: list[float], labels: list[str]) -> list[dict[str, Any]]:
    bins = [{"label": label, "count": 0} for label in labels]
    for value in values:
        for index in range(len(boundaries) - 1):
            if boundaries[index] <= value < boundaries[index + 1] or (index == len(labels) - 1 and value == boundaries[-1]):
                bins[index]["count"] += 1
                break
    return bins


def _amount_histogram(values: list[float], count: int = 5) -> list[dict[str, Any]]:
    if not values:
        return []
    low, high = min(values), max(values)
    if low == high:
        return [{"label": f"₹{low:,.0f}", "minimum": low, "maximum": high, "count": len(values)}]
    width = (high - low) / count
    bins = []
    for index in range(count):
        start = low + index * width
        end = high if index == count - 1 else low + (index + 1) * width
        bins.append({"label": f"₹{start:,.0f}–₹{end:,.0f}", "minimum": round(start, 2), "maximum": round(end, 2), "count": 0})
    for value in values:
        index = min(int((value - low) / width), count - 1)
        bins[index]["count"] += 1
    return bins


def _overview(rows: list[dict[str, str]]) -> dict[str, Any]:
    amounts = [_number(row, "requested_loan_amount") for row in rows]
    risks = Counter(row.get("risk_category", "Unknown") for row in rows)
    statuses = Counter(row.get("application_status", "Unknown") for row in rows)
    return {
        "total_loan_applications": len(rows),
        "total_requested_loan_amount": round(sum(amounts), 2),
        "average_requested_loan_amount": round(mean(amounts), 2) if amounts else 0,
        "low_risk_applications": risks["Low"],
        "medium_risk_applications": risks["Medium"],
        "high_risk_applications": risks["High"],
        "applications_under_review": statuses["Under Review"],
    }


@router.get("/overview")
def portfolio_overview() -> dict[str, Any]:
    return _overview(_records())


@router.get("/risk-distribution")
def risk_distribution() -> dict[str, Any]:
    rows = _records()
    counts = Counter(row.get("risk_category", "Unknown") for row in rows)
    return {"distribution": [{"category": category, "count": counts[category]} for category in RISK_ORDER]}


@router.get("/application-trends")
def application_trends() -> dict[str, Any]:
    monthly: dict[str, dict[str, float]] = defaultdict(lambda: {"count": 0, "requested_amount": 0.0})
    for row in _records():
        month = _month(row)
        if not month:
            continue
        monthly[month]["count"] += 1
        monthly[month]["requested_amount"] += _number(row, "requested_loan_amount")
    return {
        "trends": [
            {"month": month, "count": int(values["count"]), "requested_amount": round(values["requested_amount"], 2)}
            for month, values in sorted(monthly.items())
        ]
    }


@router.get("/eda")
def portfolio_eda() -> dict[str, Any]:
    rows = _records()
    amounts = [_number(row, "requested_loan_amount") for row in rows]
    incomes = [_number(row, "annual_income") for row in rows]
    credits = [_number(row, "credit_score") for row in rows]
    risks = Counter(row.get("risk_category", "Unknown") for row in rows)
    risk_total = sum(risks.get(category, 0) for category in RISK_ORDER)

    credit_labels = ["300–499", "500–599", "600–699", "700–799", "800–900"]
    credit_distribution = _fixed_histogram(credits, [300, 500, 600, 700, 800, 901], credit_labels)
    score_bands = [(300, 600, "300–599"), (600, 700, "600–699"), (700, 800, "700–799"), (800, 901, "800–900")]
    risk_by_credit = []
    for lower, upper, label in score_bands:
        matching = [row for row in rows if lower <= _number(row, "credit_score") < upper]
        risk_by_credit.append({
            "credit_score_band": label,
            "counts": {category: sum(row.get("risk_category") == category for row in matching) for category in RISK_ORDER},
        })

    income_bands = [(0, 500_000, "Under ₹5L"), (500_000, 1_000_000, "₹5–10L"),
                    (1_000_000, 2_500_000, "₹10–25L"), (2_500_000, 5_000_000, "₹25–50L"),
                    (5_000_000, float("inf"), "₹50L+")]
    income_vs_requested = []
    for lower, upper, label in income_bands:
        group = [row for row in rows if lower <= _number(row, "annual_income") < upper]
        group_amounts = [_number(row, "requested_loan_amount") for row in group]
        group_incomes = [_number(row, "annual_income") for row in group]
        income_vs_requested.append({
            "income_band": label,
            "application_count": len(group),
            "average_annual_income": round(mean(group_incomes), 2) if group else 0,
            "average_requested_amount": round(mean(group_amounts), 2) if group else 0,
        })

    statuses = Counter(row.get("application_status", "Unknown") for row in rows)
    employment = Counter(row.get("employment_type", "Unknown") for row in rows)
    monthly = Counter(_month(row) for row in rows if _month(row))
    sorted_months = sorted(monthly.items())
    growth = None
    if len(sorted_months) >= 2 and sorted_months[-2][1] > 0:
        growth = round((sorted_months[-1][1] - sorted_months[-2][1]) / sorted_months[-2][1] * 100, 2)

    return {
        "insights": {
            "mean_loan_amount": round(mean(amounts), 2) if amounts else 0,
            "median_loan_amount": round(median(amounts), 2) if amounts else 0,
            "mean_annual_income": round(mean(incomes), 2) if incomes else 0,
            "average_credit_score": round(mean(credits), 2) if credits else 0,
            "high_risk_percentage": round(risks.get("High", 0) / len(rows) * 100, 2) if rows else 0,
            "application_growth_percent_latest_month_over_previous": growth,
            "application_count": len(rows),
        },
        "requested_amount_distribution": _amount_histogram(amounts),
        "credit_score_distribution": credit_distribution,
        "employment_type_distribution": [{"category": key, "count": value} for key, value in sorted(employment.items())],
        "application_status_distribution": [{"category": key, "count": value} for key, value in sorted(statuses.items())],
        "income_band_vs_requested_amount": income_vs_requested,
        "risk_category_vs_credit_score": risk_by_credit,
        "risk_distribution": [{"category": category, "count": risks[category]} for category in RISK_ORDER],
    }
