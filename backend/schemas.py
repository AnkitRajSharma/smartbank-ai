"""Validated input and output schemas for the prediction APIs."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LoanApplication(StrictInput):
    age: int = Field(ge=18, le=100)
    annual_income: float = Field(gt=0, le=1_000_000_000)
    employment_type: Literal["Salaried", "Self-employed", "Contract"]
    employment_experience: float = Field(ge=0, le=80)
    credit_score: int = Field(ge=300, le=900)
    existing_loan_amount: float = Field(ge=0, le=1_000_000_000)
    monthly_debt: float = Field(ge=0, le=100_000_000)
    requested_loan_amount: float = Field(gt=0, le=1_000_000_000)


class LoanApplicationSubmission(LoanApplication):
    applicant_name: str = Field(min_length=2, max_length=120)

    @field_validator("applicant_name")
    @classmethod
    def name_must_not_be_blank(cls, value: str) -> str:
        value = " ".join(value.split())
        if len(value) < 2:
            raise ValueError("Applicant name must contain at least two characters")
        return value


class RegisterRequest(StrictInput):
    name: str = Field(min_length=2, max_length=120)
    email: str = Field(min_length=5, max_length=254)
    password: str = Field(min_length=10, max_length=128)

    @field_validator("name")
    @classmethod
    def clean_name(cls, value: str) -> str:
        value = " ".join(value.split())
        if len(value) < 2:
            raise ValueError("Name must contain at least two characters")
        return value

    @field_validator("email")
    @classmethod
    def clean_email(cls, value: str) -> str:
        value = value.strip().lower()
        if "@" not in value or value.startswith("@") or value.endswith("@"):
            raise ValueError("Enter a valid email address")
        return value


class LoginRequest(StrictInput):
    email: str = Field(min_length=5, max_length=254)
    password: str = Field(min_length=1, max_length=128)

    @field_validator("email")
    @classmethod
    def clean_email(cls, value: str) -> str:
        return value.strip().lower()


class FraudTransaction(StrictInput):
    transaction_amount: float = Field(gt=0, le=1_000_000_000)
    transaction_hour: int = Field(ge=0, le=23)
    transaction_type: Literal["Purchase", "Cash Withdrawal", "Transfer", "Online Payment"]
    merchant_category: Literal["Retail", "Groceries", "Travel", "Dining", "Electronics", "Other"]
    transactions_last_24_hours: int = Field(ge=0, le=10_000)
    distance_from_usual_location_km: float = Field(ge=0, le=50_000)
    international_transaction: bool
    transaction_channel: Literal["In Person", "Online", "Mobile", "ATM"]
