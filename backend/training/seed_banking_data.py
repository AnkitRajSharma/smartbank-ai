"""Seed deterministic synthetic account, transaction and credit data locally."""
import json
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from auth import USERS_TABLE
from json_store import JsonTable
from security import hash_password

DATA = Path(__file__).resolve().parents[1] / "data"
ACCOUNTS = JsonTable(DATA / "accounts.json")
TX = JsonTable(DATA / "transactions.json")
CREDIT = JsonTable(DATA / "credit_profiles.json")
ALERTS = JsonTable(DATA / "fraud_alerts.json")
PEOPLE = [
    {"user_id":"SB10001","name":"Aarav Sharma","email":"customer1@smartbank.demo","role":"customer","type":"Salaried","income":900000,"score":785,"category":"Excellent","opened":"2023-04-15","balance":148250,"opening_balance":100000},
    {"user_id":"SB10002","name":"Priya Verma","email":"customer2@smartbank.demo","role":"customer","type":"Self-employed","income":600000,"score":695,"category":"Fair","opened":"2022-09-10","balance":92750,"opening_balance":80000},
    {"user_id":"SB10003","name":"Rahul Mehta","email":"customer3@smartbank.demo","role":"customer","type":"Salaried","income":420000,"score":590,"category":"Poor","opened":"2024-01-20","balance":54800,"opening_balance":50000},
    {"user_id":"demo-admin-001","name":"SmartBank Analytics","email":"analytics@smartbank.demo","role":"admin"},
]
PATTERNS = [
    ("Monthly salary", "Income", "Credit", 75000), ("Fresh market groceries", "Groceries", "Debit", 2150),
    ("Electricity bill", "Utilities", "Debit", 1840), ("Online shopping", "Shopping", "Debit", 3290),
    ("ATM cash withdrawal", "Cash", "Debit", 5000), ("Restaurant payment", "Dining", "Debit", 1250),
    ("Home internet bill", "Utilities", "Debit", 999), ("Loan EMI", "Loan", "Debit", 12100),
    ("Pharmacy purchase", "Medical", "Debit", 780), ("Fuel station", "Transport", "Debit", 2300),
    ("UPI transfer received", "Transfer", "Credit", 2500), ("Retail refund", "Shopping", "Credit", 890),
]

def main():
    cred_path = DATA / "demo_credentials.json"
    if not cred_path.exists():
        raise SystemExit(f"Create {cred_path} with demo email/password pairs before seeding.")
    credentials = json.loads(cred_path.read_text(encoding="utf-8"))
    current = USERS_TABLE.read()
    for person in PEOPLE:
        if person["role"] == "admin":
            continue
        found = next((u for u in current if u.get("email", "").lower() == person["email"]), None)
        # Preserve an existing Argon2 credential when the user already signed in before seeding.
        password_hash = found.get("password_hash") if found else hash_password(credentials[person["email"]])
        if found:
            found.update({"user_id":person["user_id"], "name":person["name"], "role":"customer"})
        else:
            current.append({"user_id":person["user_id"], "name":person["name"], "email":person["email"], "role":"customer", "password_hash":password_hash})
    admin = next((u for u in current if u.get("email") == "analytics@smartbank.demo"), None)
    if admin:
        admin.update(user_id="demo-admin-001", name="SmartBank Analytics", role="admin")
    else:
        current.append({"user_id":"demo-admin-001", "name":"SmartBank Analytics", "email":"analytics@smartbank.demo", "role":"admin", "password_hash":hash_password(credentials["analytics@smartbank.demo"])})
    USERS_TABLE.replace(current)
    if not ACCOUNTS.read():
        accounts, transactions, profiles, alerts = [], [], [], []
        today = date.today()
        for idx, p in enumerate(PEOPLE[:3]):
            account_id = f"ACC-{p['user_id']}"
            monthly_income = round(p["income"] / 12)
            monthly_expenses = int(monthly_income * (0.58 + idx * .07))
            debits = credits = 0
            txs = []
            for i in range(42):
                label, category, kind, base = PATTERNS[i % len(PATTERNS)]
                amount = round(base * (1 + idx * .11 + (i % 5) * .025), 2)
                if i % len(PATTERNS) == 0: amount = monthly_income
                if kind == "Credit": credits += amount
                else: debits += amount
                d = today - timedelta(days=41-i)
                txs.append({"transaction_id":f"TX-{p['user_id']}-{i+1:03}","account_id":account_id,"date":d.isoformat(),"time":f"{8+(i*3)%13:02d}:{(i*17)%60:02d}:00","description":label,"merchant_category":category,"transaction_type":kind,"amount":amount,"channel":["UPI","Card","Online","ATM"][i%4],"status":"Completed","fraud_label":i == 29 and idx == 2,"risk_level":"Medium" if i == 29 and idx == 2 else "Low"})
            opening = round(p["balance"] - credits + debits, 2)
            accounts.append({"user_id":p["user_id"],"customer_id":p["user_id"],"name":p["name"],"email":p["email"],"account_id":account_id,"masked_account_number":f"XXXX XXXX {1001+idx}","ifsc":"SMBK0000420","account_type":"Savings","available_balance":p["balance"],"opening_balance":opening,"opening_date":p["opened"],"annual_income":p["income"],"employment_type":p["type"],"employment_experience":5+idx*3,"monthly_income":monthly_income,"monthly_expenses":monthly_expenses,"existing_loans":[] if idx == 0 else [{"type":"Personal loan","outstanding_amount":120000 if idx == 1 else 90000}],"monthly_emi":0 if idx == 0 else (10000 if idx == 1 else 7500),"loan_history":[],"synthetic_data":True})
            transactions.extend(txs)
            score = p["score"]
            profiles.append({"user_id":p["user_id"],"score":score,"category":p["category"],"utilization_percent":[18,42,67][idx],"monthly_emi":accounts[-1]["monthly_emi"],"debt_to_income_ratio":round(accounts[-1]["monthly_emi"] / monthly_income, 3),"repayment_history":"Synthetic demonstration history","insights":["Consistent on-time payment pattern in this synthetic profile." if idx == 0 else "Review repayment consistency and keep utilization manageable."]})
            if idx == 2: alerts.append({"alert_id":"FA-SB10003-001","account_id":account_id,"transaction_id":txs[29]["transaction_id"],"date":txs[29]["date"],"risk_level":"Medium","explanation":"Synthetic unusual activity example; verify activity through official channels."})
        for table, rows in [(ACCOUNTS,accounts),(TX,transactions),(CREDIT,profiles),(ALERTS,alerts)]:
            table.replace(rows)
    print("Seeded 3 synthetic customer accounts, 126 transactions, credit profiles and demo alerts.")

if __name__ == "__main__": main()
