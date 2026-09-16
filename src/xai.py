"""Explainable AI (XAI) module for RoadSentinel AI.

Combines:
1. SHAP (SHapley Additive exPlanations) for tabular feature triage models.
2. Grad-CAM (Gradient-weighted Class Activation Mapping) for visual attention explanation.
"""
from __future__ import annotations

import os
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision


# ---------------------------------------------------------------------------
# SHAP — Tabular Models
# ---------------------------------------------------------------------------

def explain_tabular_model(
    fitted_pipeline,
    X_sample: pd.DataFrame,
    save_path: str = "models/metrics/shap_summary.png",
) -> str:
    """Generate and save SHAP summary plot for a fitted pipeline."""
    import shap

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    prep = fitted_pipeline.named_steps["prep"]
    clf = fitted_pipeline.named_steps["clf"]

    X_trans = prep.transform(X_sample)
    # Get feature names from ColumnTransformer
    try:
        feature_names = prep.get_feature_names_out()
    except Exception:
        feature_names = [f"feat_{i}" for i in range(X_trans.shape[1])]

    # Convert sparse to dense if needed
    if hasattr(X_trans, "toarray"):
        X_trans = X_trans.toarray()

    plt.figure(figsize=(9, 6))
    try:
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_trans)
        if isinstance(shap_values, list) and len(shap_values) > 1:
            shap_values = shap_values[1]  # positive class (accident)
        shap.summary_plot(shap_values, X_trans, feature_names=feature_names, show=False)
    except Exception:
        # Generic explainer fallback for SVM or linear models
        explainer = shap.Explainer(clf.predict, X_trans)
        shap_values = explainer(X_trans)
        shap.summary_plot(shap_values, feature_names=feature_names, show=False)

    plt.title("RoadSentinel AI — SHAP Feature Importance Summary", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved SHAP summary plot -> {save_path}")
    return save_path


# ---------------------------------------------------------------------------
# Grad-CAM — Vision Explanations
# ---------------------------------------------------------------------------

_resnet = None


def get_vision_classifier(num_classes: int = 2, weights_path: str = "models/accident_resnet18.pth"):
    """Transfer-learned ResNet-18 backbone for Grad-CAM targeting."""
    global _resnet
    if _resnet is None:
        model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        if os.path.exists(weights_path):
            try:
                state_dict = torch.load(weights_path, map_location="cpu")
                model.load_state_dict(state_dict)
                print(f"[XAI] Loaded trained ResNet-18 weights from {weights_path}")
            except Exception as e:
                print(f"[XAI] Notice: Could not load trained weights ({e}), using default head.")
        model.eval()
        _resnet = model
    return _resnet



def explain_frame_gradcam(
    frame_bgr: np.ndarray,
    save_path: str | None = None,
) -> np.ndarray:
    """Generate Grad-CAM visual attention overlay on an incident frame."""
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image

    model = get_vision_classifier()
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)

    # Preprocess BGR -> RGB and normalize for ResNet
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    norm_img = rgb.astype(np.float32) / 255.0

    # ImageNet standard normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    tensor_img = (norm_img - mean) / std
    tensor_img = torch.tensor(tensor_img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

    try:
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        targets = [ClassifierOutputTarget(1)]
        grayscale_cam = cam(input_tensor=tensor_img, targets=targets)[0]
    except Exception:
        grayscale_cam = cam(input_tensor=tensor_img, targets=None)[0]

    overlay = show_cam_on_image(norm_img, grayscale_cam, use_rgb=True)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # Convert RGB back to BGR for OpenCV saving
        cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print(f"Saved Grad-CAM overlay -> {save_path}")

    return overlay
