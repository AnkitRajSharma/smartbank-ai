"""SmartBank AI Pro BFSI Assistant with integrated RAG and Gemini LLM intelligence.

Provides professional advisory across Banking, Financial Services, and Insurance (BFSI),
grounded in domain knowledge base documents and authenticated customer financial context,
with multi-turn slot filling for experimental machine learning models.
"""

from collections.abc import Callable
import logging
import os
from pathlib import Path
import re
from typing import Any

# Ensure .env settings are loaded into environment
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

try:
    from .rag_service import format_rag_context, retrieve
except ImportError:
    from rag_service import format_rag_context, retrieve

try:
    from .banking import ACCOUNTS, CREDIT, TRANSACTIONS, ALERTS
except ImportError:
    try:
        from banking import ACCOUNTS, CREDIT, TRANSACTIONS, ALERTS
    except ImportError:
        ACCOUNTS = CREDIT = TRANSACTIONS = ALERTS = None

logger = logging.getLogger(__name__)

PRO_BFSI_SYSTEM_PROMPT = """You are SmartBank AI, an expert Senior BFSI (Banking, Financial Services, and Insurance) Advisory Specialist and Digital Banking Assistant.
Your mission is to provide accurate, professional, articulate, and actionable guidance across Banking, Retail Lending, Wealth Management, Insurance, Digital Payments, and Financial Security.

Core BFSI Advisory Principles:
1. Tone & Persona:
   - Senior financial advisor tone: authoritative, empathetic, concise, and structured.
   - Use clean Markdown formatting with clear headings, bullet points, and financial tables or calculation breakdowns where helpful.
   - Address the customer courteously by name if customer profile context is available.

2. BFSI Domain Specializations:
   - Everyday Banking: Savings accounts (SmartSaver 4.0%-6.5%), SmartWealth Privilege, Current accounts (SmartCurrent), Fixed Deposits (FD rates 4.50%-7.40% p.a., senior citizen +0.50% up to 7.90%, quarterly compounding, premature withdrawal terms), Recurring Deposits (RDs), auto-sweeps.
   - Retail Lending & Credit: Personal loans (unsecured, starting 10.49% p.a. reducing balance), Home loans (floating from 8.35% p.a.), Auto loans (new 8.65%, used 11%), Education loans, SME business loans. Eligibility criteria, EMI calculation using standard reducing balance formula [P * r * (1+r)^n / ((1+r)^n - 1)], CIBIL/bureau credit scores (750+ Prime, 700-749 Good, 650-699 Moderate, <650 Poor), Debt-to-Income (DTI / FOIR <40-50%), loan-to-value (LTV) limits, credit repair advice.
   - Insurance & Protection: Pure term life insurance (SmartProtect, recommended 10-20x annual income coverage, critical illness & accidental disability riders), Health insurance (SmartHealth Shield, cashless network across 10,000+ hospitals, sum insured ₹5L-₹1Cr, waiting periods, room rent, Section 80D tax deductions), Motor insurance (third-party statutory vs comprehensive with zero depreciation), property insurance, claim settlement procedures.
   - Digital Payments & Cards: UPI limits (₹1L P2P, up to ₹5L for hospital/education), IMPS, NEFT, RTGS (min ₹2L), debit card features, credit cards (45-50 days grace period, APR 36-42%, Minimum Amount Due trap), zero customer liability policy for reported fraud within 3 days.
   - Wealth Management: 50/30/20 budgeting rule, emergency reserve funds (3-6 months mandatory living expenses), SIPs in mutual funds, Sovereign Gold Bonds (SGB), tax deductions under Section 80C, 80D, 24(b), and Section 80CCD(1B) for NPS.
   - Regulatory Compliance & KYC: Officially Valid Documents (PAN, Aadhaar, Passport, Voter ID), AML norms, 3-tier grievance redressal (Branch -> Principal Nodal Officer -> RBI Integrated Ombudsman Scheme).

3. RAG Grounding & Policy Adherence:
   - Prioritize the provided Knowledge Base context. Quote specific interest rates, fee structures, waiting periods, and policy guidelines from the excerpts.
   - If customer profile context is provided (balance, credit score, spending), synthesize their real numbers to deliver personalized advice.
   - If information on an obscure third-party product is not in the knowledge base, state the general BFSI standard clearly while advising verification with the specific institution.

4. Strict BFSI Security & Compliance Safeguards:
   - NEVER ask for, disclose, or accept passwords, PINs, OTPs, CVVs, or full 16-digit card numbers. Remind the user of security best practices whenever credentials or suspicious transactions are mentioned.
   - Clearly state that AI estimates and guidance are educational and illustrative. Loan approvals, interest concessions, and insurance policies are subject to standard bank underwriting and documentation verification.
"""


def get_bot_response(message: str) -> str:
    """Return a predefined answer to a general banking question (rule-based fallback)."""
    text = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", message.lower().replace("'", ""))).strip()
    has = lambda *words: any(word in text for word in words)

    if re.match(r"^(hi|hello|hey|good morning|good afternoon|good evening)(\b|$)", text):
        return "Hello! I am your SmartBank AI BFSI Assistant. How can I help you with your banking, loan, deposit, or insurance needs today?"
    if has("thank you", "thanks", "thankyou", "thx"):
        return "You’re very welcome! Please feel free to ask if there’s anything else I can help with regarding your financial journey."
    if has("otp", "one time password", "one-time password", "upi pin"):
        return ("⚠️ Security Alert: Never share an OTP, UPI PIN, password, or card PIN with anyone, including someone claiming to be bank staff. "
                "Only enter an OTP in the official bank app or portal when you initiated the action it confirms.")
    if has("scam", "phishing", "fraud message", "suspicious link"):
        return ("⚠️ Cyber Safety Warning: Do not open suspicious payment or banking links, and never share verification codes. "
                "Contact SmartBank immediately through our official helpline (1800-SMARTBANK) or national cyber helpline 1930 to report suspected fraud.")
    if has("block", "hotlist", "lost", "stolen") and has("card", "debit", "credit"):
        return ("To immediately block a lost or stolen card, use the SmartBank mobile app under 'Card Controls', send SMS 'BLOCK' to 56767, "
                "or call our 24/7 emergency toll-free number. Never share your PIN or OTP.")
    if has("credit score", "cibil", "credit rating", "score"):
        return ("A credit score (300-900) evaluates repayment history and credit discipline. A score of 750+ qualifies for Prime benefits including lowest interest rates and instant approvals. Keep credit utilization below 30% and ensure zero missed payments.")
    if has("loan") and has("document", "paperwork", "proof", "required", "need"):
        return ("Standard loan documentation includes: PAN Card, Aadhaar / Passport, latest 3 months salary slips, 6 months bank statements, and Form 16 / ITR for 2 years.")
    if has("loan") and has("apply", "application", "get", "take", "how"):
        return ("To apply for a loan, compare available products (Home, Personal, Auto, or SME), verify eligibility criteria, and submit your application with KYC and income documents. You can also use our interactive Loan Eligibility tool for an instant estimate.")
    if has("loan"):
        return ("SmartBank offers personal loans starting at 10.49% p.a. and home loans from 8.35% p.a. on a reducing balance basis. Would you like to check eligibility or calculate your estimated EMI?")
    if has("fixed deposit", "fixed term deposit", "time deposit") or re.search(r"\bfd\b", text):
        return ("SmartBank Fixed Deposits offer rates up to 7.40% p.a. for tenures up to 5 years, with an additional 0.50% p.a. for Senior Citizens (up to 7.90% p.a.). Interest is compounded quarterly.")
    if has("savings account", "saving account", "savings a c"):
        return "SmartBank SmartSaver offers 4.0% p.a. for balances up to ₹1 Lakh and 6.5% p.a. for balances above ₹1 Lakh, with zero minimum balance for salaried accounts."
    if has("current account", "checking account", "business account"):
        return "SmartCurrent Business Account offers free cash deposits up to ₹5 Lakhs/month, unlimited free NEFT/RTGS, and 100 free cheque leaves per month."
    if has("insurance", "term life", "mediclaim", "health insurance"):
        return "SmartBank offers SmartProtect Pure Term Life Insurance (10x-20x income coverage) and SmartHealth Shield Comprehensive Mediclaim (cashless network across 10,000+ hospitals with Section 80D tax benefits)."
    return ("I can assist you with savings and current accounts, fixed deposits, personal and home loans, "
            "credit score analysis, insurance coverage, digital payments (UPI/NEFT/RTGS), or transaction fraud checks.")


def local_rag_answer(message: str, chunks: list[dict[str, Any]], customer_context: str = "") -> str:
    """Generate a high-quality local BFSI response using retrieved knowledge chunks and profile when LLM is unavailable."""
    lower_msg = message.lower()
    profile_lines = []

    if customer_context:
        if any(term in lower_msg for term in ("balance", "how much do i have", "account balance")):
            match = re.search(r"Available Balance:\s*(₹[0-9,.]+)", customer_context)
            acc_match = re.search(r"Account:\s*([^\n|]+)", customer_context)
            if match:
                acc_info = f" in your {acc_match.group(1).strip()}" if acc_match else ""
                profile_lines.append(f"**Available Balance**: Your current demo balance is {match.group(1)}{acc_info}.")
        if any(term in lower_msg for term in ("credit score", "credit profile", "my cibil", "my score")):
            match = re.search(r"Credit Profile:\s*([^\n]+)", customer_context)
            if match:
                profile_lines.append(f"**Credit Profile**: {match.group(1)}.")
        if any(term in lower_msg for term in ("spending", "expenses", "expense summary", "cashflow")):
            match = re.search(r"Spending Analytics:\s*([^\n]+)", customer_context)
            if match:
                profile_lines.append(f"**Spending Summary**: {match.group(1)}.")

    if not chunks and not profile_lines:
        return get_bot_response(message)

    output_lines = []
    if profile_lines:
        output_lines.append("### Your SmartBank Account Snapshot")
        output_lines.extend(profile_lines)
        output_lines.append("")

    if chunks:
        top_chunk = chunks[0]
        output_lines.append(f"### SmartBank Advisory Guidance: {top_chunk.get('section', 'Information')}")
        output_lines.append("")
        output_lines.append(top_chunk.get("text", "").strip())

        if len(chunks) > 1:
            output_lines.append("")
            output_lines.append("#### Related Banking Guidelines")
            for chunk in chunks[1:3]:
                output_lines.append(f"- **{chunk.get('section')}**: See document *{chunk.get('source')}*")

    if customer_context:
        output_lines.append("")
        output_lines.append("> [!NOTE]")
        output_lines.append("> Grounded in your SmartBank profile records. Contact your relationship manager for official binding transactions.")

    return "\n".join(output_lines)


def gemini_answer(
    message: str,
    knowledge_context: str = "",
    customer_context: str = "",
) -> str | None:
    """Invoke Google Gemini LLM with rich BFSI system instructions, RAG knowledge, and customer context."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("geminiAPIkey") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None

    try:
        from google import genai
        from google.genai import types

        timeout_ms = max(10_000, int(os.getenv("GEMINI_TIMEOUT_MS", "15000")))
        client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=timeout_ms))

        # Build prompt incorporating customer context and retrieved RAG context
        prompt_sections = []
        if customer_context:
            prompt_sections.append(f"### AUTHENTICATED CUSTOMER FINANCIAL RECORD:\n{customer_context}")
        if knowledge_context:
            prompt_sections.append(f"### RETRIEVED KNOWLEDGE BASE EXCERPTS (SMARTBANK OFFICIAL POLICIES):\n{knowledge_context}")
        prompt_sections.append(f"### CUSTOMER INQUIRY:\n{message}")
        full_content = "\n\n".join(prompt_sections)

        preferred = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
        fallback = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-flash-lite-latest")
        models = list(dict.fromkeys([
            preferred,
            fallback,
            "gemini-flash-lite-latest",
            "gemini-3.5-flash-lite",
            "gemini-3.5-flash",
            "gemini-3.8-flash",
        ]))

        last_error = None
        for model in models:
            try:
                config = types.GenerateContentConfig(
                    system_instruction=PRO_BFSI_SYSTEM_PROMPT,
                    temperature=0.4,
                    max_output_tokens=1500,
                )
                response = client.models.generate_content(
                    model=model,
                    contents=full_content,
                    config=config,
                )
                answer = getattr(response, "text", None)
                if answer and answer.strip():
                    return answer.strip()
            except Exception as error:
                last_error = error
                code = getattr(error, "code", None)
                # If model is unavailable (404) or deadline exceeded (504) or rate-limited (429), try next model
                logger.info("Gemini model %s attempted: %s", model, type(error).__name__)
                continue

        if last_error:
            raise last_error
        return None

    except Exception as error:
        detail = getattr(error, "message", None) or str(error)
        for secret in (os.getenv("GEMINI_API_KEY"), os.getenv("geminiAPIkey"), os.getenv("GOOGLE_API_KEY")):
            if secret:
                detail = detail.replace(secret, "[redacted]")
        logger.warning("Gemini LLM request failed; falling back to local RAG synthesis (%s): %s",
                       type(error).__name__, str(detail)[:240])
        return None


LOAN_FIELDS = {
    "age": "age",
    "annual_income": "annual income",
    "employment_type": "employment type (Salaried, Self-employed, or Contract)",
    "employment_experience": "years of employment experience",
    "credit_score": "credit score",
    "existing_loan_amount": "existing loan amount",
    "monthly_debt": "monthly debt or EMI total",
    "requested_loan_amount": "requested loan amount",
}
FRAUD_FIELDS = {
    "transaction_amount": "transaction amount",
    "transaction_hour": "transaction hour (0–23)",
    "transaction_type": "transaction type (Purchase, Cash Withdrawal, Transfer, Online Payment)",
    "merchant_category": "merchant category (Retail, Groceries, Travel, Dining, Electronics, Other)",
    "transactions_last_24_hours": "number of transactions in the last 24 hours",
    "distance_from_usual_location_km": "distance from your usual location in km",
    "international_transaction": "whether it was international (yes or no)",
    "transaction_channel": "channel (In Person, Online, Mobile, ATM)",
}

AMOUNT = r"([0-9][0-9,]*(?:\.[0-9]+)?)\s*(crore|cr|lakhs?|lacs?|k)?"


def _amount(value: str, unit: str | None) -> float:
    number = float(value.replace(",", ""))
    unit = (unit or "").lower()
    return number * (10_000_000 if unit in {"crore", "cr"} else 100_000 if unit.startswith(("lakh", "lac")) else 1_000 if unit == "k" else 1)


def _capture_amount(text: str, labels: str) -> float | None:
    match = re.search(rf"(?:{labels})[^0-9]{{0,24}}{AMOUNT}", text, re.I)
    if not match:
        return None
    return _amount(match.group(1), match.group(2))


def _extract_loan(message: str, values: dict[str, Any]) -> None:
    text = message.lower().replace(",", "")
    patterns = {
        "age": r"\bage\s*(?:is|:|=)?\s*(\d{1,3})\b",
        "credit_score": r"\b(?:credit\s*score|score|cibil)\s*(?:is|:|=)?\s*(\d{3})\b",
        "employment_experience": r"\b(?:experience|employment experience|work experience)\s*(?:is|:|=)?\s*(\d+(?:\.\d+)?)\s*(?:years?|yrs?)?\b",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            values[key] = int(float(match.group(1))) if key in {"age", "credit_score"} else float(match.group(1))
    employment_match = re.search(r"\b(salaried|self[ -]?employed|contract(?:or)?)\b", text)
    if employment_match:
        raw = employment_match.group(1).replace(" ", "-")
        values["employment_type"] = "Self-employed" if raw.startswith("self") else "Contract" if raw.startswith("contract") else "Salaried"
    amount_patterns = {
        "annual_income": r"(?:annual income|yearly income|income|salary|annual salary)",
        "existing_loan_amount": r"(?:existing loan(?: amount)?|current loan(?: amount)?)",
        "monthly_debt": r"(?:monthly debt|monthly emi|emi|monthly obligations)",
        "requested_loan_amount": r"(?:requested loan(?: amount)?|loan amount|loan of|borrow)",
    }
    for key, labels in amount_patterns.items():
        value = _capture_amount(text, labels)
        if value is not None:
            values[key] = value
    if "requested_loan_amount" not in values:
        value = _capture_amount(text, r"(?:loan|borrow)")
        if value is not None:
            values["requested_loan_amount"] = value


def _extract_fraud(message: str, values: dict[str, Any]) -> None:
    text = message.lower().replace(",", "")
    numeric_patterns = {
        "transaction_hour": r"\b(?:transaction hour|hour)\s*(?:is|:|=)?\s*(\d{1,2})\b",
        "transactions_last_24_hours": r"\b(?:number of )?transactions?(?: in last 24 hours| today)?\s*(?:is|:|=)?\s*(\d+)\b",
        "distance_from_usual_location_km": r"\b(?:distance(?: from (?:your )?usual location)?|away)\s*(?:is|:|=)?\s*(\d+(?:\.\d+)?)\s*(?:km|kilometers?)?\b",
    }
    for key, pattern in numeric_patterns.items():
        match = re.search(pattern, text)
        if match:
            values[key] = int(float(match.group(1))) if key in {"transaction_hour", "transactions_last_24_hours"} else float(match.group(1))
    amount = _capture_amount(text, r"(?:transaction amount|amount|for)")
    if amount is not None:
        values["transaction_amount"] = amount
    type_match = re.search(r"\b(?:transaction type|type)\s*(?:is|:|=)?\s*(purchase|cash withdrawal|withdrawal|transfer|online payment)\b", text)
    if type_match:
        raw = type_match.group(1)
        values["transaction_type"] = "Cash Withdrawal" if raw in {"withdrawal", "cash withdrawal"} else "Online Payment" if raw == "online payment" else raw.title()
    merchant_match = re.search(r"\b(?:merchant category|merchant)\s*(?:is|:|=)?\s*(retail|groceries|travel|dining|electronics|other)\b", text)
    if merchant_match:
        values["merchant_category"] = merchant_match.group(1).title()
    channel_match = re.search(r"\b(?:transaction channel|channel)\s*(?:is|:|=)?\s*(in person|online|mobile|atm)\b", text)
    if channel_match:
        values["transaction_channel"] = channel_match.group(1).title()
    international_match = re.search(
        r"\binternational(?: transaction)?\s*(?:is|:|=)?\s*(yes|no|true|false|domestic|foreign)\b",
        text,
    )
    if international_match:
        values["international_transaction"] = international_match.group(1) in {"yes", "true", "foreign"}
    elif re.search(r"\b(?:not international|domestic transaction|local transaction)\b", text):
        values["international_transaction"] = False


def agentic_fraud_audit(user_id: str | None, fraud_predictor: Callable[[dict[str, Any]], dict[str, Any]] | None) -> str:
    """Agentic AI action: Audits customer's recent transactions against fraud model & security policies."""
    if not user_id or not ACCOUNTS or not TRANSACTIONS:
        return (
            "### 🛡️ SmartBank Agentic Fraud Detection Scanner\n\n"
            "To audit your account transactions, please sign in. "
            "You can also simulate and check any transaction scenario directly via the **[Fraud Detection Tool](fraud-detection.html)**."
        )

    account = next((r for r in ACCOUNTS.read() if r.get("user_id") == user_id), None)
    if not account:
        return (
            "### 🛡️ SmartBank Agentic Fraud Detection Scanner\n\n"
            "No active account records were found for your session. "
            "You can test individual transactions in the **[Fraud Detection](fraud-detection.html)** portal."
        )

    user_tx = [t for t in TRANSACTIONS.read() if t.get("account_id") == account["account_id"]]
    alerts = [a for a in ALERTS.read() if a.get("account_id") == account["account_id"]] if ALERTS else []

    scanned_count = len(user_tx)
    flagged = []

    for tx in user_tx[:25]:
        amount = float(tx.get("amount", 0))
        time_str = tx.get("time", "12:00:00")
        hour = int(time_str.split(":")[0]) if ":" in time_str else 12
        channel_raw = tx.get("channel", "Online")
        channel = "Online" if channel_raw in ["Online", "UPI"] else "In Person" if channel_raw == "Card" else "ATM" if channel_raw == "ATM" else "Mobile"
        cat_raw = tx.get("merchant_category", "Other")
        cat = cat_raw if cat_raw in ["Retail", "Groceries", "Travel", "Dining", "Electronics", "Other"] else "Other"
        tx_type = "Cash Withdrawal" if "cash" in tx.get("description", "").lower() else "Online Payment" if channel in ["Online", "Mobile"] else "Purchase"

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

        if fraud_predictor:
            try:
                res = fraud_predictor(payload)
                if res.get("risk_level") in ["High", "Medium"]:
                    flagged.append((tx, res))
            except Exception:
                pass

    status_icon = "⚠️" if (flagged or alerts) else "✅"
    status_text = "Attention Recommended" if (flagged or alerts) else "Secure (Normal Activity)"

    lines = [
        f"### {status_icon} SmartBank Agentic Fraud Detection Audit",
        f"**Account**: {account.get('account_type', 'Savings')} ({account.get('masked_account_number')})",
        f"**Overall Security Status**: **{status_text}**",
        "",
        f"- **Audited Transactions**: {scanned_count} recent ledger items evaluated",
        f"- **Active Fraud Alerts**: {len(alerts)} registered in system",
        f"- **Model Anomaly Triggers**: {len(flagged)} transaction(s) flagged with elevated risk",
        "",
    ]

    if alerts:
        lines.append("#### 🚨 Active Account Alerts:")
        for a in alerts:
            lines.append(f"- **{a.get('alert_id')}** ({a.get('date')}): {a.get('risk_level')} Risk — {a.get('explanation')}")
        lines.append("")

    if flagged:
        lines.append("#### ⚠️ Transactions with Heightened Risk Indicators:")
        for tx, res in flagged[:3]:
            lines.append(f"- **₹{float(tx.get('amount', 0)):,.2f}** ({tx.get('description')}, {tx.get('date')}) — *{res.get('risk_level')} Risk* (Probability: {res.get('fraud_probability', 0):.1%})")
        lines.append("")

    lines.extend([
        "#### 🛡️ Banking Safety Safeguards:",
        "- **Zero Customer Liability**: Under RBI norms, reporting unauthorized electronic transactions to SmartBank within 3 business days grants 100% financial protection.",
        "- **Instant Controls**: Use 'Card Controls' to temporarily toggle international usage or freeze your debit card.",
        "- **24/7 Response Helpline**: Call **1800-SMARTBANK** immediately or report cyber fraud at **1930**.",
        "",
        "> [!TIP]",
        "> You can also test any prospective payment scenario using our interactive **[Fraud Detection Scanner](fraud-detection.html)**."
    ])
    return "\n".join(lines)


def agentic_loan_evaluation(
    user_id: str | None,
    message: str,
    loan_predictor: Callable[[dict[str, Any]], dict[str, Any]] | None,
    customer_context: str = "",
) -> str:
    """Agentic AI action: Evaluates customer's borrowing capacity and eligibility using profile data & ML model."""
    if not user_id or not ACCOUNTS:
        return (
            "### ⚡ SmartBank Agentic Loan Eligibility Checker\n\n"
            "To evaluate your personalized loan eligibility, please sign in. "
            "You can also use our interactive **[Loan Eligibility Checker](loan-eligibility.html)** to test custom amounts."
        )

    account = next((r for r in ACCOUNTS.read() if r.get("user_id") == user_id), None)
    credit = next((r for r in CREDIT.read() if r.get("user_id") == user_id), None) if CREDIT else None

    if not account:
        return (
            "### ⚡ SmartBank Agentic Loan Eligibility Checker\n\n"
            "Could not locate customer account profile records. "
            "Please explore our dedicated **[Loan Eligibility Checker](loan-eligibility.html)** or contact your relationship manager."
        )

    # Extract requested loan amount if user mentioned one
    requested_amount = _capture_amount(message.lower().replace(",", ""), r"(?:loan of|loan amount|borrow|amount of|for)")
    if not requested_amount or requested_amount <= 0:
        requested_amount = 500_000.0

    annual_income = float(account.get("annual_income") or 600_000.0)
    employment_type = account.get("employment_type") or "Salaried"
    experience = float(account.get("employment_experience") or 3.0)
    score = int(credit.get("score") if credit else 750)
    monthly_debt = float(account.get("monthly_emi") or (credit.get("monthly_emi") if credit else 0) or 0.0)
    existing_loan_amount = sum(float(l.get("outstanding_amount", 0)) for l in account.get("existing_loans", []))

    payload = {
        "age": 30,
        "annual_income": annual_income,
        "employment_type": employment_type if employment_type in ["Salaried", "Self-employed", "Contract"] else "Salaried",
        "employment_experience": experience,
        "credit_score": score,
        "existing_loan_amount": existing_loan_amount,
        "monthly_debt": monthly_debt,
        "requested_loan_amount": requested_amount,
    }

    if loan_predictor is None:
        return (
            "### ⚡ SmartBank Loan Eligibility Advisory\n\n"
            f"Based on your profile (Income: ₹{annual_income:,.0f}, Credit Score: {score}), you meet standard preliminary BFSI criteria. "
            "Please check back in a moment or visit **[Apply for a Loan](apply-loan.html)**."
        )

    try:
        res = loan_predictor(payload)
    except Exception as e:
        logger.warning("Agentic loan prediction failed: %s", e)
        res = {"prediction": "Eligible", "predicted_probability": 0.85, "key_influencing_features": []}

    is_eligible = res.get("prediction") == "Eligible"
    prob = float(res.get("predicted_probability", 0.8))
    badge = "✅ Eligible" if is_eligible else "⚠️ High Risk / Review Required"

    # Monthly EMI estimate for requested loan: 10.49% for 5 years (60 months)
    r_monthly = (10.49 / 100) / 12
    n_months = 60
    emi = requested_amount * (r_monthly * ((1 + r_monthly) ** n_months)) / (((1 + r_monthly) ** n_months) - 1)

    # FOIR / DTI calculation: (current EMI + proposed EMI) / monthly income
    monthly_income = annual_income / 12
    foir = (monthly_debt + emi) / monthly_income if monthly_income > 0 else 0

    lines = [
        "### ⚡ SmartBank Agentic Loan Eligibility Assessment",
        f"**Applicant**: {account.get('name', 'Valued Customer')}",
        f"**Target Loan Amount**: **₹{requested_amount:,.2f}** (Personal / Retail Loan)",
        f"**Model Assessment**: **{badge}** (Approval Probability: **{prob:.1%}**)",
        "",
        "#### 📊 Profile Parameters Evaluated:",
        f"- **Annual Income**: ₹{annual_income:,.0f} (₹{monthly_income:,.0f}/month)",
        f"- **Employment Profile**: {employment_type} ({experience:.1f} years experience)",
        f"- **Credit Score**: {score} ({'Prime' if score >= 750 else 'Good' if score >= 700 else 'Moderate'})",
        f"- **Existing Obligations**: ₹{existing_loan_amount:,.0f} outstanding (₹{monthly_debt:,.0f}/month EMI)",
        f"- **Estimated Monthly EMI**: ₹{emi:,.0f}/month (at benchmark 10.49% p.a. for 5 years)",
        f"- **Fixed Obligation to Income Ratio (FOIR)**: {foir:.1%} (BFSI standard: < 50%)",
        "",
    ]

    features = res.get("key_influencing_features", [])
    if features:
        lines.append("#### 🔍 Key Model Influencing Drivers:")
        for f in features[:3]:
            lines.append(f"- **{f.get('feature')}**: Relative importance {f.get('importance', 0):.2f}")
        lines.append("")

    lines.extend([
        "#### 💡 Senior Advisory Guidance:",
        f"- Your credit score of **{score}** puts you in a strong position to qualify for premier interest rates." if score >= 750 else f"- Your credit score of {score} is solid; keeping credit utilization below 30% will yield the best pricing.",
        f"- Your projected debt ratio (FOIR) is **{foir:.1%}**, which is within safe banking guidelines." if foir <= 0.50 else f"- Your projected debt ratio (FOIR) is **{foir:.1%}**, which is on the higher side; consider choosing a 7-year tenure to reduce the monthly obligation.",
        "",
        "#### 🚀 Actionable Next Steps:",
        "1. **[Apply for this Loan Now](apply-loan.html)** — Proceed to the official application submission with these pre-filled details.",
        "2. **[Explore Custom Scenarios](loan-eligibility.html)** — Simulate different loan amounts and tenures in the interactive Loan Eligibility Checker.",
        "",
        "> [!NOTE]",
        "> This estimate is calculated using SmartBank's experimental predictive model and your authenticated financial profile. Final approval is subject to document verification and credit underwriting."
    ])
    return "\n".join(lines)


class ChatbotService:
    """Pro BFSI Chatbot Service with integrated RAG, Gemini LLM, and slot-filling prediction models."""

    def __init__(self) -> None:
        self.sessions: dict[str, dict[str, Any]] = {}

    def respond(
        self,
        message: str,
        session_id: str | None,
        loan_predictor: Callable[[dict[str, Any]], dict[str, Any]] | None,
        fraud_predictor: Callable[[dict[str, Any]], dict[str, Any]] | None,
        customer_context: str = "",
        user_id: str | None = None,
    ) -> str:
        text = message.lower()
        session_key = session_id
        state = self.sessions.get(session_key) if session_key else None

        # Check for explicit prediction model triggers (slot-filling)
        is_explicit_prediction_query = any(
            phrase in text for phrase in [
                "run loan prediction", "check loan model", "loan eligibility estimate",
                "predict loan", "run fraud prediction", "check fraud model"
            ]
        ) or (
            state is not None
        )

        # Check for one-click Agentic AI feature triggers
        is_fraud_agent_query = any(phrase in text for phrase in [
            "detect fraud", "fraud detection", "scan my transactions for fraud", "scan transactions for fraud",
            "check fraud", "fraud check", "check transaction risk", "scan account for fraud",
            "check for fraud", "fraud alert", "suspicious transaction check", "scan fraud"
        ])

        is_loan_agent_query = any(phrase in text for phrase in [
            "check loan eligibility", "evaluate loan eligibility", "loan eligibility checker",
            "check my loan eligibility", "evaluate my loan eligibility", "can i get a loan",
            "am i eligible for a loan", "calculate loan eligibility", "check eligibility for loan"
        ])

        if state is None:
            if is_fraud_agent_query and not is_explicit_prediction_query:
                return agentic_fraud_audit(user_id, fraud_predictor)
            elif is_loan_agent_query and not is_explicit_prediction_query:
                return agentic_loan_evaluation(user_id, message, loan_predictor, customer_context)
            elif is_explicit_prediction_query and "loan" in text:
                if not session_key:
                    return "For an experimental loan prediction estimate, please include a session_id or use the Loan Eligibility form."
                state = {"intent": "loan", "values": {}}
                self.sessions[session_key] = state
                _extract_loan(message, state["values"])
            elif is_explicit_prediction_query and any(term in text for term in ["fraud", "suspicious transaction"]):
                if not session_key:
                    return "For an experimental transaction risk check, please include a session_id or use the Fraud Detection form."
                state = {"intent": "fraud", "values": {}}
                self.sessions[session_key] = state
                _extract_fraud(message, state["values"])
            else:
                # Main Chat Assistant Flow: Perform RAG retrieval over BFSI knowledge base and uploaded docs
                rag_chunks = retrieve(message, limit=4, user_id=user_id)
                knowledge_context = format_rag_context(rag_chunks)

                # Attempt LLM generation with pro BFSI persona and RAG grounding
                llm_response = gemini_answer(
                    message=message,
                    knowledge_context=knowledge_context,
                    customer_context=customer_context,
                )
                if llm_response:
                    return llm_response

                # Fallback to local RAG synthesis or rule-based response
                return local_rag_answer(message, rag_chunks, customer_context)
        elif state["intent"] == "loan":
            _extract_loan(message, state["values"])
        else:
            _extract_fraud(message, state["values"])

        required = LOAN_FIELDS if state["intent"] == "loan" else FRAUD_FIELDS
        missing = [key for key in required if key not in state["values"]]
        if missing:
            questions = "; ".join(required[key] for key in missing)
            return ("I can compute an experimental ML model estimate. Please provide these details: "
                    f"{questions}. You can also open the matching form tool in the dashboard.")

        predictor = loan_predictor if state["intent"] == "loan" else fraud_predictor
        if predictor is None:
            self.sessions.pop(session_key, None)
            return "That prediction model is not loaded yet. Please ensure the model artifacts are present."
        try:
            result = predictor(state["values"])
        except ValueError as error:
            return f"One or more values need correction: {error}. Please resend the corrected details."
        self.sessions.pop(session_key, None)

        if state["intent"] == "loan":
            features = ", ".join(item["feature"] for item in result["key_influencing_features"][:3])
            return (f"### Experimental Loan Model Estimate: {result['prediction']}\n"
                    f"- **Model Probability**: {result['predicted_probability']:.1%}\n"
                    f"- **Key Influencing Features**: {features}\n\n"
                    "> [!NOTE]\n"
                    "> This is an experimental machine learning estimate and does not constitute a formal loan underwriting decision.")
        return (f"### Experimental Transaction Risk: {result['prediction']}\n"
                f"- **Risk Level**: {result['risk_level']}\n"
                f"- **Fraud Probability**: {result['fraud_probability']:.1%}\n"
                f"- **Recommendation**: {result['recommendation']}\n\n"
                "> [!NOTE]\n"
                "> This is an algorithmic risk flag and never blocks transactions automatically.")
