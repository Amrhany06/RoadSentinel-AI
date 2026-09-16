"""Master Training & Artifact Generation Pipeline for RoadSentinel AI.

Orchestrates complete training across all modules:
1. Classical Triage Classifiers (SVM, RF, LogReg, XGBoost) + Unified Metrics
2. Severity K-Means Clustering + Silhouette Validation Curve
3. EMS Response Time Regressor (RandomForest) + Residual Diagnostics
4. Geospatial Hotspot Clustering (K-Means + PCA) + Folium Interactive Map
5. Explainable AI: SHAP Summary & Grad-CAM Heatmap
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")

# Ensure root directory is on python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import joblib
import pandas as pd

from scripts.seed_data_generator import generate_all
from src.clustering import find_best_k, render_hotspot_map, train_severity_kmeans, train_spatial_hotspots
from src.models import (
    evaluate_and_compare_models,
    train_logreg,
    train_rf,
    train_svm,
    train_xgb,
)
from src.preprocessing import split_first
from src.regression import evaluate_and_plot_regression, split_regression_data, train_rf_regressor
from src.xai import explain_frame_gradcam, explain_tabular_model


def run_training_pipeline(data_dir: str = "data", models_dir: str = "models", demo_dir: str = "demo"):
    print("=" * 60)
    print("      ROADSENTINEL AI — MASTER TRAINING & ARTIFACT PIPELINE      ")
    print("=" * 60)

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(os.path.join(models_dir, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(demo_dir, "precomputed", "gradcam"), exist_ok=True)

    features_path = os.path.join(data_dir, "engineered_features.csv")
    ems_path = os.path.join(data_dir, "nyc_ems_response.csv")
    spatial_path = os.path.join(data_dir, "us_accidents_sample.csv")

    # If seed datasets are missing, generate them automatically
    if not (os.path.exists(features_path) and os.path.exists(ems_path) and os.path.exists(spatial_path)):
        print("\n[Bootstrap] Data assets missing. Generating realistic seed data...")
        generate_all()

    # -------------------------------------------------------------
    # 1. CLASSICAL TRIAGE CLASSIFIERS
    # -------------------------------------------------------------
    print("\n--- [1/5] Training Classical ML Triage Classifiers ---")
    df_features = pd.read_csv(features_path)
    X_train, X_test, y_train, y_test = split_first(df_features, label_col="is_accident")
    print(f"Dataset split: Train={len(X_train)} samples, Test={len(X_test)} samples (Accident rate={y_train.mean():.2%})")

    print("Fitting SVM with GridSearchCV...")
    svm_pipe = train_svm(X_train, y_train)

    print("Fitting Random Forest...")
    rf_pipe = train_rf(X_train, y_train)

    print("Fitting Logistic Regression baseline...")
    logreg_pipe = train_logreg(X_train, y_train)

    print("Fitting XGBoost Classifier...")
    xgb_pipe = train_xgb(X_train, y_train)

    models_dict = {
        "SVM": svm_pipe,
        "RandomForest": rf_pipe,
        "LogisticRegression": logreg_pipe,
        "XGBoost": xgb_pipe,
    }

    # Save primary deployment model
    joblib.dump(svm_pipe, os.path.join(models_dir, "svm_triage_pipeline.joblib"))
    joblib.dump(rf_pipe, os.path.join(models_dir, "rf_triage_pipeline.joblib"))
    joblib.dump(xgb_pipe, os.path.join(models_dir, "xgb_triage_pipeline.joblib"))

    metrics_df = evaluate_and_compare_models(models_dict, X_test, y_test, save_dir=os.path.join(models_dir, "metrics"))
    print("\nTriage Model Leaderboard:")
    print(metrics_df[["Model", "Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC", "PR-AUC"]].to_string(index=False))

    # -------------------------------------------------------------
    # 2. UNSUPERVISED INCIDENT CLUSTERING
    # -------------------------------------------------------------
    print("\n--- [2/5] Training Incident Severity K-Means Clusterer ---")
    num_cols = ["avg_speed", "max_speed", "max_deceleration", "trajectory_variance", "max_iou"]
    flagged_features = df_features[df_features["is_accident"] == 1][num_cols]
    best_k, _ = find_best_k(
        flagged_features,
        k_range=range(2, 6),
        save_plot_path=os.path.join(models_dir, "metrics", "silhouette_curve.png"),
    )
    kmeans_model = train_severity_kmeans(flagged_features, k=best_k)
    joblib.dump(kmeans_model, os.path.join(models_dir, "severity_kmeans.joblib"))
    print(f"Saved severity K-Means model (k={best_k}) -> {models_dir}/severity_kmeans.joblib")

    # -------------------------------------------------------------
    # 3. EMS RESPONSE TIME REGRESSION
    # -------------------------------------------------------------
    print("\n--- [3/5] Training Emergency Response Time Regressor ---")
    ems_df = pd.read_csv(ems_path)
    X_reg_train, X_reg_test, y_reg_train, y_reg_test = split_regression_data(ems_df)
    rf_reg = train_rf_regressor(X_reg_train, y_reg_train)
    evaluate_and_plot_regression(
        rf_reg,
        X_reg_test,
        y_reg_test,
        save_path=os.path.join(models_dir, "metrics", "regression_metrics.png"),
    )
    joblib.dump(rf_reg, os.path.join(models_dir, "response_time_regressor.joblib"))
    print(f"Saved response time regressor -> {models_dir}/response_time_regressor.joblib")

    # -------------------------------------------------------------
    # 4. SPATIAL HOTSPOT CLUSTERING & FOLIUM MAP
    # -------------------------------------------------------------
    print("\n--- [4/5] Training Geospatial Accident Hotspot Clustering ---")
    spatial_df = pd.read_csv(spatial_path)
    hotspot_kmeans, _, _, labeled_coords, _ = train_spatial_hotspots(
        spatial_df,
        k_range=range(3, 7),
        save_pca_plot=os.path.join(models_dir, "metrics", "spatial_pca_clusters.png"),
    )
    joblib.dump(hotspot_kmeans, os.path.join(models_dir, "hotspot_kmeans.joblib"))
    render_hotspot_map(labeled_coords, out_path=os.path.join(demo_dir, "hotspot_map.html"))

    # -------------------------------------------------------------
    # 5. EXPLAINABLE AI (SHAP + GRAD-CAM)
    # -------------------------------------------------------------
    print("\n--- [5/5] Generating Explainable AI (XAI) Artifacts ---")
    try:
        sample_X = X_test.sample(min(150, len(X_test)), random_state=42)
        explain_tabular_model(xgb_pipe, sample_X, save_path=os.path.join(models_dir, "metrics", "shap_summary.png"))
    except Exception as e:
        print(f"[XAI Note] SHAP calculation deferred: {e}")

    # Generate sample Grad-CAM heatmap on demo frame
    sample_frame = os.path.join(demo_dir, "clip_1_frame.jpg")
    if os.path.exists(sample_frame):
        frame_bgr = cv2.imread(sample_frame)
        explain_frame_gradcam(
            frame_bgr,
            save_path=os.path.join(demo_dir, "precomputed", "gradcam", "clip_1_gradcam.png"),
        )

    print("\n" + "=" * 60)
    print("      ALL MODELS & EVALUATION ARTIFACTS SUCCESSFULLY GENERATED      ")
    print("=" * 60)


if __name__ == "__main__":
    run_training_pipeline()
