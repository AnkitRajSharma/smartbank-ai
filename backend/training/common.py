"""Shared utilities for reproducible synthetic demonstration datasets and reports."""

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
MODELS = ROOT.parent / "models"
REPORTS = ROOT / "reports"
MODELS.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)


def make_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric, numeric_features),
        ("cat", categorical, categorical_features),
    ])


def save_json(data: dict[str, Any], destination: Path) -> None:
    destination.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def save_confusion_plot(matrix: list[list[int]], labels: list[str], destination: Path, title: str) -> None:
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(destination, dpi=150)
    plt.close()


def save_comparison_plot(metrics: dict[str, dict[str, float]], destination: Path, title: str) -> None:
    frame = pd.DataFrame(metrics).T
    columns = [name for name in ["precision", "recall", "f1", "roc_auc", "pr_auc_average_precision"] if name in frame.columns]
    frame[columns].plot(kind="bar", figsize=(8, 4), ylim=(0, 1), rot=0)
    plt.title(title)
    plt.ylabel("Validation metric")
    plt.tight_layout()
    plt.savefig(destination, dpi=150)
    plt.close()


def save_eda(data: pd.DataFrame, target: str, destination: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    sns.countplot(data=data, x=target, ax=axes[0], color="#176b87")
    axes[0].set_title("Target class distribution")
    axes[0].set_xlabel(target)
    numeric = [column for column in data.select_dtypes(include="number").columns if column != target]
    if numeric:
        sns.histplot(data=data, x=numeric[0], bins=24, ax=axes[1], color="#37b6a5")
        axes[1].set_title(f"Distribution: {numeric[0]}")
    else:
        axes[1].set_visible(False)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(destination, dpi=150)
    plt.close(fig)
