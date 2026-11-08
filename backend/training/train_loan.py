"""Train, compare, evaluate, and save the experimental loan classifier."""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import MODELS, REPORTS, make_preprocessor, save_comparison_plot, save_confusion_plot, save_eda, save_json

SEED = 42
DATA_PATH = Path(__file__).resolve().parent / "loan_dataset.csv"
TARGET = "eligible"
NUMERIC = ["age", "annual_income", "employment_experience", "credit_score",
           "existing_loan_amount", "monthly_debt", "requested_loan_amount"]
CATEGORICAL = ["employment_type"]
FEATURES = NUMERIC + CATEGORICAL


def generate_synthetic_dataset(path: Path = DATA_PATH, rows: int = 6000) -> pd.DataFrame:
    """Create a deterministic teaching dataset from explicit illustrative rules."""
    rng = np.random.default_rng(SEED)
    age = rng.integers(21, 71, rows)
    annual_income = np.clip(rng.lognormal(np.log(900_000), 0.72, rows), 180_000, 12_000_000)
    employment = rng.choice(["Salaried", "Self-employed", "Contract"], rows, p=[.58, .30, .12])
    experience = np.clip(rng.normal((age - 21) * .48, 5.0), 0, 45).round(1)
    credit = np.clip(rng.normal(690, 82, rows), 300, 900).round().astype(int)
    existing = np.clip(rng.lognormal(np.log(180_000), 1.0, rows) - 70_000, 0, 3_500_000)
    existing[rng.random(rows) < .32] = 0
    monthly_debt = np.clip(existing / rng.uniform(18, 72, rows) + rng.lognormal(8.0, .85, rows), 0, 250_000)
    requested = np.clip(rng.lognormal(np.log(650_000), .85, rows), 50_000, 8_000_000)

    debt_ratio = (monthly_debt + requested / 48) / (annual_income / 12)
    logit = (
        2.2 + (credit - 650) / 58 - debt_ratio * 3.2
        + (experience - 4) * .035
        + np.where(employment == "Salaried", .32, np.where(employment == "Contract", -.28, 0))
        - requested / annual_income * .65
        - np.maximum(age - 62, 0) * .035
    )
    probability = 1 / (1 + np.exp(-np.clip(logit, -12, 12)))
    eligible = (rng.random(rows) < probability).astype(int)

    frame = pd.DataFrame({
        "age": age,
        "annual_income": annual_income.round(2),
        "employment_type": employment,
        "employment_experience": experience,
        "credit_score": credit,
        "existing_loan_amount": existing.round(2),
        "monthly_debt": monthly_debt.round(2),
        "requested_loan_amount": requested.round(2),
        TARGET: eligible,
    })
    # A small, reproducible missing-data rate exercises the training imputation pipeline.
    for column in ["annual_income", "employment_type", "monthly_debt"]:
        frame.loc[rng.random(rows) < .012, column] = np.nan
    frame.to_csv(path, index=False)
    return frame


def metrics(y_true: pd.Series, predicted: np.ndarray, probabilities: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
    }


def main() -> None:
    data = pd.read_csv(DATA_PATH) if DATA_PATH.exists() else generate_synthetic_dataset()
    data = data.dropna(subset=[TARGET])
    save_eda(data, TARGET, REPORTS / "loan_eda.png", "Synthetic loan dataset EDA")
    x = data[FEATURES]
    y = data[TARGET].astype(int)
    x_train, x_temp, y_train, y_temp = train_test_split(
        x, y, test_size=.4, stratify=y, random_state=SEED
    )
    x_validation, x_test, y_validation, y_test = train_test_split(
        x_temp, y_temp, test_size=.5, stratify=y_temp, random_state=SEED
    )

    candidates = {
        "LogisticRegression": LogisticRegression(max_iter=1500, class_weight="balanced", random_state=SEED),
        "DecisionTree": DecisionTreeClassifier(max_depth=8, min_samples_leaf=15, class_weight="balanced", random_state=SEED),
        "RandomForest": RandomForestClassifier(n_estimators=250, min_samples_leaf=5, class_weight="balanced_subsample", n_jobs=-1, random_state=SEED),
    }
    fitted = {}
    validation_metrics = {}
    for name, estimator in candidates.items():
        pipeline = Pipeline([("preprocessor", make_preprocessor(NUMERIC, CATEGORICAL)), ("model", estimator)])
        pipeline.fit(x_train, y_train)
        fitted[name] = pipeline
        validation_metrics[name] = metrics(y_validation, pipeline.predict(x_validation), pipeline.predict_proba(x_validation)[:, 1])

    selected_name = max(validation_metrics, key=lambda name: (validation_metrics[name]["f1"], validation_metrics[name]["roc_auc"]))
    # Refit on train + validation only after selection; the test partition remains untouched.
    selected = Pipeline([("preprocessor", make_preprocessor(NUMERIC, CATEGORICAL)),
                         ("model", candidates[selected_name])])
    selected.fit(pd.concat([x_train, x_validation]), pd.concat([y_train, y_validation]))
    test_prediction = selected.predict(x_test)
    test_probability = selected.predict_proba(x_test)[:, 1]
    test_metrics = metrics(y_test, test_prediction, test_probability)
    matrix = confusion_matrix(y_test, test_prediction, labels=[0, 1]).tolist()
    joblib.dump(selected, MODELS / "loan_pipeline.pkl")
    report = {
        "dataset": "synthetic illustrative data; not sourced from bank applications",
        "assumptions": [
            "Labels are sampled from a hand-designed probability rule using credit score, debt burden, income, requested amount, experience, employment type, and age.",
            "The rule and distributions are for software demonstration only and do not represent a lender's underwriting policy.",
            "A small amount of missing data is inserted to exercise imputation.",
        ],
        "seed": SEED,
        "rows": int(len(data)),
        "target_distribution": {str(k): int(v) for k, v in data[TARGET].value_counts().sort_index().items()},
        "split": {"train": len(x_train), "validation": len(x_validation), "test": len(x_test)},
        "validation_model_comparison": validation_metrics,
        "selected_model": selected_name,
        "test_metrics": test_metrics,
        "test_confusion_matrix_labels_0_1": matrix,
        "test_confusion_matrix": {"true_negative": matrix[0][0], "false_positive": matrix[0][1], "false_negative": matrix[1][0], "true_positive": matrix[1][1]},
        "feature_columns": FEATURES,
        "test_set_used_for_model_selection": False,
    }
    save_json(report, REPORTS / "loan_model_report.json")
    save_confusion_plot(matrix, ["Not Eligible", "Eligible"], REPORTS / "loan_confusion_matrix.png", "Loan model test confusion matrix")
    save_comparison_plot(validation_metrics, REPORTS / "loan_validation_comparison.png", "Loan model validation comparison")
    print(json.dumps({"selected_model": selected_name, "test_metrics": test_metrics, "confusion_matrix": matrix}, indent=2))


if __name__ == "__main__":
    main()
