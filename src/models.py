"""Classical-ML triage classifiers for RoadSentinel AI.

Trains and evaluates leak-safe sklearn Pipelines (preprocessing + classifier)
for SVM, Random Forest, Logistic Regression, and XGBoost.
Saves unified ROC, PR, and Confusion Matrix plots to models/metrics/.
"""
from __future__ import annotations

import json
import os
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.preprocessing import build_preprocessor


def train_svm(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Train an SVM classifier inside a leak-safe Pipeline with GridSearchCV."""
    pipe = Pipeline([
        ("prep", build_preprocessor()),
        ("clf", SVC(probability=True, random_state=42)),
    ])
    grid = GridSearchCV(
        pipe,
        param_grid={
            "clf__C": [0.5, 1.0, 5.0],
            "clf__kernel": ["rbf", "linear"],
        },
        cv=3,
        scoring="f1",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    print("Best SVM parameters:", grid.best_params_)
    return grid.best_estimator_


def train_rf(X_train: pd.DataFrame, y_train: pd.Series, class_weight: str | None = "balanced") -> Pipeline:
    """Train a Random Forest classifier with class weighting."""
    pipe = Pipeline([
        ("prep", build_preprocessor()),
        ("clf", RandomForestClassifier(n_estimators=200, class_weight=class_weight, random_state=42, n_jobs=-1)),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def train_logreg(X_train: pd.DataFrame, y_train: pd.Series, class_weight: str | None = "balanced") -> Pipeline:
    """Train a Logistic Regression baseline classifier."""
    pipe = Pipeline([
        ("prep", build_preprocessor()),
        ("clf", LogisticRegression(class_weight=class_weight, max_iter=1000, random_state=42)),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def train_xgb(X_train: pd.DataFrame, y_train: pd.Series, scale_pos_weight: float | None = None) -> Pipeline:
    """Train an XGBoost gradient-boosted classifier with scale_pos_weight."""
    if scale_pos_weight is None:
        neg = int((y_train == 0).sum())
        pos = int((y_train == 1).sum())
        scale_pos_weight = float(neg / max(1, pos))

    pipe = Pipeline([
        ("prep", build_preprocessor()),
        ("clf", XGBClassifier(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.06,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def evaluate_single_model(name: str, pipe: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """Compute headline performance metrics for one pipeline."""
    preds = pipe.predict(X_test)
    probs = pipe.predict_proba(X_test)[:, 1]

    prec, rec, _ = precision_recall_curve(y_test, probs)
    pr_auc = auc(rec, prec)

    metrics = {
        "Model": name,
        "Accuracy": float(accuracy_score(y_test, preds)),
        "Precision": float(precision_score(y_test, preds, zero_division=0)),
        "Recall": float(recall_score(y_test, preds, zero_division=0)),
        "F1 Score": float(f1_score(y_test, preds, zero_division=0)),
        "ROC-AUC": float(roc_auc_score(y_test, probs)),
        "PR-AUC": float(pr_auc),
    }
    return metrics


def evaluate_and_compare_models(
    models: Dict[str, Pipeline],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save_dir: str = "models/metrics",
) -> pd.DataFrame:
    """Generate and save unified ROC curves, PR curves, Confusion Matrices, and leaderboard CSV."""
    os.makedirs(save_dir, exist_ok=True)
    summary_records = []

    # 1. Multi-Model ROC Curve
    plt.figure(figsize=(8, 6))
    for name, pipe in models.items():
        probs = pipe.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, probs)
        score = roc_auc_score(y_test, probs)
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {score:.3f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Guess")
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity / Recall)")
    plt.title("RoadSentinel AI — Triage Model ROC Curves")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    roc_path = os.path.join(save_dir, "roc_curves_comparison.png")
    plt.savefig(roc_path, dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Multi-Model Precision-Recall Curve
    plt.figure(figsize=(8, 6))
    for name, pipe in models.items():
        probs = pipe.predict_proba(X_test)[:, 1]
        prec, rec, _ = precision_recall_curve(y_test, probs)
        score = auc(rec, prec)
        plt.plot(rec, prec, lw=2, label=f"{name} (PR-AUC = {score:.3f})")

    baseline = y_test.mean()
    plt.axhline(y=baseline, color="k", linestyle="--", lw=1.5, label=f"Baseline Pos Rate ({baseline:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("RoadSentinel AI — Precision-Recall Curves (Imbalance Triage)")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    pr_path = os.path.join(save_dir, "pr_curves_comparison.png")
    plt.savefig(pr_path, dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Confusion Matrices 2x2 Grid
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()
    for idx, (name, pipe) in enumerate(models.items()):
        preds = pipe.predict(X_test)
        cm = confusion_matrix(y_test, preds)
        ax = axes[idx]
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.set_title(f"{name} Confusion Matrix")
        fig.colorbar(im, ax=ax)
        tick_marks = np.arange(2)
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(["Normal", "Accident"])
        ax.set_yticklabels(["Normal", "Accident"])

        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j, i, format(cm[i, j], "d"),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black"
                )
        ax.set_ylabel("True label")
        ax.set_xlabel("Predicted label")

        # Collect record
        summary_records.append(evaluate_single_model(name, pipe, X_test, y_test))

    plt.tight_layout()
    cm_path = os.path.join(save_dir, "confusion_matrices.png")
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()

    # 4. Save Leaderboard CSV & JSON
    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(os.path.join(save_dir, "metrics_summary.csv"), index=False)
    with open(os.path.join(save_dir, "metrics_summary.json"), "w") as f:
        json.dump(summary_records, f, indent=2)

    print(f"Generated evaluation suite saved to {save_dir}/:")
    print(" - roc_curves_comparison.png")
    print(" - pr_curves_comparison.png")
    print(" - confusion_matrices.png")
    print(" - metrics_summary.csv")
    return summary_df
