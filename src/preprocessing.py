"""Leak-safe preprocessing.

The rule this module exists to enforce: split first, fit only on the training
split, and do both inside a single scikit-learn object so it is structurally
impossible to leak test-set statistics into training by accident.

OneHotEncoder (not LabelEncoder) is used for feature columns, since LabelEncoder
imposes a false ordinal relationship (e.g. "fog" > "clear") that hurts
distance- or coefficient-based models like SVM and Logistic Regression.
LabelEncoder is the right tool for encoding a *target* column, not features.
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

CAT_COLS = ["road_type", "weather", "time_of_day"]
NUM_COLS = ["avg_speed", "max_speed", "max_deceleration", "trajectory_variance", "max_iou"]
ALL_FEATURE_COLS = CAT_COLS + NUM_COLS


def split_first(df: pd.DataFrame, label_col: str = "is_accident", test_size: float = 0.2, random_state: int = 42):
    """Always split BEFORE fitting anything on the data.

    Returns X_train, X_test, y_train, y_test with raw (unencoded, unscaled)
    feature values — the ColumnTransformer built by build_preprocessor()
    handles encoding/scaling downstream, fit only on X_train.
    """
    missing = [c for c in ALL_FEATURE_COLS + [label_col] if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing expected columns: {missing}")

    X = df[ALL_FEATURE_COLS]
    y = df[label_col]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def build_preprocessor() -> ColumnTransformer:
    """A fresh, unfit ColumnTransformer. Never call .fit() on this directly in
    application code — wrap it in a sklearn Pipeline with the model (see
    src/models.py) so preprocessing and model always fit/transform together."""
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
            ("num", MinMaxScaler(), NUM_COLS),
        ]
    )
