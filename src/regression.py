"""Regression module for predicting emergency response time (minutes).

Trained on EMS/911 incident dispatch features (borough, time_of_day,
incident_severity_level, call_volume_density).
"""
from __future__ import annotations

import os
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

REG_CAT_COLS = ["borough", "time_of_day", "incident_severity_level"]
REG_NUM_COLS = ["call_volume_density"]


def build_regression_preprocessor() -> ColumnTransformer:
    """Preprocessor for EMS tabular features."""
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), REG_CAT_COLS),
        ("num", MinMaxScaler(), REG_NUM_COLS),
    ])


def split_regression_data(ems_df: pd.DataFrame, target_col: str = "response_minutes") -> Tuple:
    """Split regression data into train and test sets."""
    X = ems_df[REG_CAT_COLS + REG_NUM_COLS]
    y = ems_df[target_col]
    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_ridge(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Train Ridge regressor with alpha tuning."""
    pipe = Pipeline([
        ("prep", build_regression_preprocessor()),
        ("reg", Ridge()),
    ])
    grid = GridSearchCV(pipe, param_grid={"reg__alpha": [0.1, 1.0, 10.0]}, cv=3, scoring="r2")
    grid.fit(X_train, y_train)
    print("Best Ridge alpha:", grid.best_params_)
    return grid.best_estimator_


def train_rf_regressor(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Train Random Forest regressor."""
    pipe = Pipeline([
        ("prep", build_regression_preprocessor()),
        ("reg", RandomForestRegressor(n_estimators=250, max_depth=8, random_state=42, n_jobs=-1)),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def evaluate_and_plot_regression(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save_path: str = "models/metrics/regression_metrics.png",
) -> Dict[str, float]:
    """Report R2, MAE, RMSE and save prediction scatter and residual plots."""
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Scatter: Actual vs Predicted
    ax1.scatter(y_test, preds, alpha=0.4, color="royalblue", edgecolor="none")
    lims = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
    ax1.plot(lims, lims, "r--", lw=2, label="Perfect Fit (y = x)")
    ax1.set_xlabel("Actual Response Time (minutes)")
    ax1.set_ylabel("Predicted Response Time (minutes)")
    ax1.set_title(f"Actual vs. Predicted (R² = {r2:.3f}, MAE = {mae:.2f}m)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Residuals plot
    residuals = y_test - preds
    ax2.hist(residuals, bins=30, color="darkorange", edgecolor="black", alpha=0.7)
    ax2.axvline(0, color="black", linestyle="--", lw=1.5)
    ax2.set_xlabel("Prediction Residuals (Actual - Predicted)")
    ax2.set_ylabel("Frequency")
    ax2.set_title(f"Residual Distribution (RMSE = {rmse:.2f}m)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    metrics = {"R2": r2, "MAE": mae, "RMSE": rmse}
    print("Regression Evaluation:", metrics)
    return metrics
