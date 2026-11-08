"""Create local demo accounts and synthetic example applications in JSON files."""

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from applications import APPLICATIONS_TABLE, risk_category
from auth import USERS_TABLE
from loan_model import load_pipeline, predict_loan
from security import hash_password


DEMO_USERS = [
    {"user_id": "SB10001", "name": "Aarav Sharma", "email": "customer1@smartbank.demo", "role": "customer", "password_env": "SMARTBANK_DEMO_CUSTOMER1_PASSWORD"},
    {"user_id": "SB10002", "name": "Priya Verma", "email": "customer2@smartbank.demo", "role": "customer", "password_env": "SMARTBANK_DEMO_CUSTOMER2_PASSWORD"},
    {"user_id": "SB10003", "name": "Rahul Mehta", "email": "customer3@smartbank.demo", "role": "customer", "password_env": "SMARTBANK_DEMO_CUSTOMER3_PASSWORD"},
    {"user_id": "demo-admin-001", "name": "SmartBank Analytics", "email": "analytics@smartbank.demo", "role": "admin", "password_env": "SMARTBANK_DEMO_ADMIN_PASSWORD"},
]

PROFILES = [
    {"age": 34, "annual_income": 1_200_000, "employment_type": "Salaried", "employment_experience": 8, "credit_score": 740, "existing_loan_amount": 100_000, "monthly_debt": 15_000, "requested_loan_amount": 500_000},
    {"age": 42, "annual_income": 850_000, "employment_type": "Self-employed", "employment_experience": 12, "credit_score": 675, "existing_loan_amount": 220_000, "monthly_debt": 24_000, "requested_loan_amount": 750_000},
    {"age": 29, "annual_income": 620_000, "employment_type": "Contract", "employment_experience": 5, "credit_score": 610, "existing_loan_amount": 350_000, "monthly_debt": 32_000, "requested_loan_amount": 900_000},
    {"age": 51, "annual_income": 2_400_000, "employment_type": "Salaried", "employment_experience": 22, "credit_score": 790, "existing_loan_amount": 0, "monthly_debt": 18_000, "requested_loan_amount": 1_100_000},
    {"age": 37, "annual_income": 1_050_000, "employment_type": "Self-employed", "employment_experience": 10, "credit_score": 705, "existing_loan_amount": 160_000, "monthly_debt": 17_000, "requested_loan_amount": 640_000},
    {"age": 46, "annual_income": 780_000, "employment_type": "Contract", "employment_experience": 15, "credit_score": 645, "existing_loan_amount": 280_000, "monthly_debt": 29_000, "requested_loan_amount": 820_000},
]


def prior_month(year: int, month: int, months_ago: int) -> tuple[int, int]:
    absolute = year * 12 + month - 1 - months_ago
    return absolute // 12, absolute % 12 + 1


def main() -> None:
    missing = [user["password_env"] for user in DEMO_USERS if not os.getenv(user["password_env"])]
    if missing:
        raise SystemExit("Set demo passwords in these environment variables before seeding: " + ", ".join(missing))
    for details in DEMO_USERS:
        user = {key: value for key, value in details.items() if key != "password_env"}
        user.update(password_hash=hash_password(os.environ[details["password_env"]]), created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        if not USERS_TABLE.append_unique(user, "email"):
            print(f"Kept existing account: {user['email']}")

    existing_ids = {row.get("application_id") for row in APPLICATIONS_TABLE.read()}
    pipeline = load_pipeline()
    now = datetime.now(timezone.utc)
    created = 0
    for index, (profile, customer) in enumerate(zip(PROFILES, DEMO_USERS[:3])):
        for offset, multiplier in enumerate((0.72, 1.0)):
            values = dict(profile)
            values["requested_loan_amount"] *= multiplier
            prediction = predict_loan(pipeline, values)
            probability = float(prediction["predicted_probability"])
            score, category = risk_category(probability)
            application_id = f"SB-DEMO-{index + 1:02d}-{offset + 1:02d}"
            if application_id in existing_ids:
                continue
            year, month = prior_month(now.year, now.month, index * 2 + offset)
            record = {
                "application_id": application_id,
                "user_id": customer["user_id"],
                "applicant_name": customer["name"],
                **values,
                "loan_probability": probability,
                "risk_score": score,
                "risk_category": category,
                "prediction": prediction["prediction"],
                "application_status": "Under Review",
                "application_date": datetime(year, month, 1, 10, 0, tzinfo=timezone.utc).isoformat(timespec="seconds"),
            }
            APPLICATIONS_TABLE.append(record)
            created += 1
    print(f"Seeded demo accounts and {created} synthetic loan applications.")


if __name__ == "__main__":
    main()
