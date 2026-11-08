"""Train, compare, evaluate, and save the experimental fraud classifier."""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import MODELS, REPORTS, make_preprocessor, save_comparison_plot, save_confusion_plot, save_eda, save_json

SEED = 43
DATA_PATH = Path(__file__).resolve().parent / "fraud_dataset.csv"
TARGET = "is_fraud"
NUMERIC = ["transaction_amount", "transaction_hour", "transactions_last_24_hours", "distance_from_usual_location_km"]
CATEGORICAL = ["transaction_type", "merchant_category", "international_transaction", "transaction_channel"]
FEATURES = NUMERIC + CATEGORICAL


def generate_synthetic_dataset(path: Path = DATA_PATH, rows: int = 12000) -> pd.DataFrame:
    """Create a reproducible rare-class example using explicit synthetic risk rules."""
    rng = np.random.default_rng(SEED)
    international = rng.random(rows) < .12
    amount = np.clip(rng.lognormal(np.log(1_500), 1.15, rows), 50, 500_000)
    amount[international] *= rng.uniform(1.3, 3.2, international.sum())
    hour = rng.integers(0, 24, rows)
    transaction_type = rng.choice(["Purchase", "Cash Withdrawal", "Transfer", "Online Payment"], rows,
                                  p=[.36, .12, .22, .30])
    merchant = rng.choice(["Retail", "Groceries", "Travel", "Dining", "Electronics", "Other"], rows,
                          p=[.24, .22, .12, .17, .10, .15])
    count = np.clip(rng.poisson(2.0, rows), 0, 30)
    distance = np.where(international, rng.gamma(3.5, 1800, rows), rng.gamma(1.4, 17, rows))
    channel = rng.choice(["In Person", "Online", "Mobile", "ATM"], rows, p=[.30, .29, .31, .10])

    score = (
        -4.0 + international * 1.55 + (hour < 5) * .9 + (hour >= 23) * .55
        + (distance > 500) * 1.25 + (count >= 7) * 1.05 + (amount > 50_000) * 1.05
        + np.isin(transaction_type, ["Transfer", "Cash Withdrawal"]) * .35
        + np.isin(channel, ["Online", "ATM"]) * .25
    )
    probability = 1 / (1 + np.exp(-score))
    fraud = (rng.random(rows) < probability).astype(int)
    frame = pd.DataFrame({
        "transaction_amount": amount.round(2),
        "transaction_hour": hour,
        "transaction_type": transaction_type,
        "merchant_category": merchant,
        "transactions_last_24_hours": count,
        "distance_from_usual_location_km": distance.round(2),
        "international_transaction": international,
        "transaction_channel": channel,
        TARGET: fraud,
    })
    for column in ["transaction_amount", "merchant_category", "distance_from_usual_location_km"]:
        frame.loc[rng.random(rows) < .008, column] = np.nan
    frame.to_csv(path, index=False)
    return frame


def metrics(y_true: pd.Series, predicted: np.ndarray, probabilities: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "pr_auc_average_precision": float(average_precision_score(y_true, probabilities)),
    }


def main() -> None:
    data = pd.read_csv(DATA_PATH) if DATA_PATH.exists() else generate_synthetic_dataset()
    data = data.dropna(subset=[TARGET])
    save_eda(data, TARGET, REPORTS / "fraud_eda.png", "Synthetic transaction dataset EDA")
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
        "RandomForest": RandomForestClassifier(n_estimators=250, min_samples_leaf=4, class_weight="balanced_subsample", n_jobs=-1, random_state=SEED),
    }
    validation_metrics = {}
    for name, estimator in candidates.items():
        pipeline = Pipeline([("preprocessor", make_preprocessor(NUMERIC, CATEGORICAL)), ("model", estimator)])
        pipeline.fit(x_train, y_train)  # class weights affect training only
        validation_metrics[name] = metrics(y_validation, pipeline.predict(x_validation), pipeline.predict_proba(x_validation)[:, 1])

    selected_name = max(validation_metrics, key=lambda name: (validation_metrics[name]["f1"], validation_metrics[name]["pr_auc_average_precision"]))
    selected = Pipeline([("preprocessor", make_preprocessor(NUMERIC, CATEGORICAL)),
                         ("model", candidates[selected_name])])
    selected.fit(pd.concat([x_train, x_validation]), pd.concat([y_train, y_validation]))
    test_prediction = selected.predict(x_test)
    test_probability = selected.predict_proba(x_test)[:, 1]
    test_metrics = metrics(y_test, test_prediction, test_probability)
    matrix = confusion_matrix(y_test, test_prediction, labels=[0, 1]).tolist()

    # Save representative false positives and false negatives for manual inspection.
    errors = x_test.copy()
    errors["actual_is_fraud"] = y_test
    errors["predicted_is_fraud"] = test_prediction
    errors["predicted_fraud_probability"] = test_probability.round(4)
    errors["error_type"] = np.select(
        [(y_test.to_numpy() == 0) & (test_prediction == 1),
         (y_test.to_numpy() == 1) & (test_prediction == 0)],
        ["false_positive", "false_negative"], default="correct"
    )
    errors[errors.error_type != "correct"].head(200).to_csv(REPORTS / "fraud_test_errors.csv", index=False)

    joblib.dump(selected, MODELS / "fraud_pipeline.pkl")
    report = {
        "dataset": "synthetic illustrative data; not sourced from real payment networks",
        "assumptions": [
            "Fraud labels are sampled from a hand-designed probability rule using transaction amount, hour, velocity, distance, international status, type, and channel.",
            "The generated class prevalence and feature distributions do not reflect a known institution or payment network.",
            "Class weights are used in the model fit on the training partition only; validation and test sets retain their natural generated distribution.",
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
        "false_positive_count": int(((y_test.to_numpy() == 0) & (test_prediction == 1)).sum()),
        "false_negative_count": int(((y_test.to_numpy() == 1) & (test_prediction == 0)).sum()),
        "test_error_examples_file": "fraud_test_errors.csv",
        "feature_columns": FEATURES,
        "test_set_used_for_model_selection": False,
    }
    save_json(report, REPORTS / "fraud_model_report.json")
    save_confusion_plot(matrix, ["Normal", "Fraud"], REPORTS / "fraud_confusion_matrix.png", "Fraud model test confusion matrix")
    save_comparison_plot(validation_metrics, REPORTS / "fraud_validation_comparison.png", "Fraud model validation comparison")
    print(json.dumps({"selected_model": selected_name, "test_metrics": test_metrics, "confusion_matrix": matrix}, indent=2))


if __name__ == "__main__":
    main()
