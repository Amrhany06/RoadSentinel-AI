"""Feature engineering: turn raw per-frame tracked bounding boxes into the
motion/overlap features the classical-ML triage layer (src/models.py) uses.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def engineer_motion_features(
    tracks_df: pd.DataFrame,
    fps: int = 30,
    width: float = 852.0,
    height: float = 480.0,
) -> pd.DataFrame:
    """Per-track calibrated physical motion features derived from bounding box trajectories.

    Applies rolling-average smoothing to eliminate detector noise jitter,
    scales pixel displacements into physical meters using calibrated camera FOV (28m reference),
    and calculates accurate velocity in km/h and peak deceleration in m/s².

    Returns one row per track_id with: avg_speed, max_speed, max_deceleration,
    trajectory_variance, duration_frames.
    """
    if tracks_df.empty:
        return pd.DataFrame(
            columns=["track_id", "avg_speed", "max_speed", "max_deceleration", "trajectory_variance", "duration_frames"]
        )

    feats = []
    L_ref = 28.0  # reference roadway field-of-view in meters
    width = max(1.0, float(width))
    height = max(1.0, float(height))
    fps = max(1.0, float(fps))

    for tid, g in tracks_df.groupby("track_id"):
        if len(g) < 2:
            continue
        g = g.sort_values("frame")

        # Smooth raw bounding-box coordinates to eliminate single-frame detector jitter
        if len(g) >= 4:
            x_smooth = g["x"].rolling(window=3, min_periods=1, center=True).mean()
            y_smooth = g["y"].rolling(window=3, min_periods=1, center=True).mean()
        else:
            x_smooth = g["x"]
            y_smooth = g["y"]

        # Normalized physical displacement in meters
        dx = (x_smooth.diff() / width) * L_ref
        dy = (y_smooth.diff() / height) * L_ref
        dt = (g["frame"].diff() / fps).replace(0, np.nan)

        speed_ms = np.sqrt(dx**2 + dy**2) / dt
        speed_kmh = (speed_ms * 3.6).clip(lower=0.0, upper=160.0)
        accel_ms2 = speed_ms.diff() / dt

        # Deceleration is negative acceleration (m/s²); use 5th percentile to eliminate glitch spikes
        valid_accel = accel_ms2.dropna()
        if len(valid_accel) >= 4:
            max_decel = float(valid_accel.quantile(0.05))
            max_spd = float(speed_kmh.quantile(0.95))
        else:
            max_decel = float(valid_accel.min()) if len(valid_accel) else 0.0
            max_spd = float(speed_kmh.max(skipna=True)) if len(speed_kmh.dropna()) else 0.0

        feats.append(
            {
                "track_id": tid,
                "avg_speed": float(speed_kmh.mean(skipna=True)),
                "max_speed": max_spd,
                "max_deceleration": max_decel,
                "trajectory_variance": float(g[["x", "y"]].var().sum()) / (width * height),
                "duration_frames": len(g),
            }
        )
    return pd.DataFrame(feats).fillna(0)


def _iou(b1, b2) -> float:
    """Intersection-over-union of two (x_center, y_center, w, h) boxes."""
    xa = max(b1[0] - b1[2] / 2, b2[0] - b2[2] / 2)
    ya = max(b1[1] - b1[3] / 2, b2[1] - b2[3] / 2)
    xb = min(b1[0] + b1[2] / 2, b2[0] + b2[2] / 2)
    yb = min(b1[1] + b1[3] / 2, b2[1] + b2[3] / 2)
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    a1, a2 = b1[2] * b1[3], b2[2] * b2[3]
    return inter / (a1 + a2 - inter + 1e-9)


def iou_overlap_features(tracks_df: pd.DataFrame) -> pd.DataFrame:
    """Max pairwise bounding-box overlap per frame — a cheap collision proxy.

    Returns columns: frame, max_iou.
    """
    if tracks_df.empty:
        return pd.DataFrame(columns=["frame", "max_iou"])

    rows = []
    for frame, g in tracks_df.groupby("frame"):
        boxes = g[["x", "y", "w", "h"]].values
        best = 0.0
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                best = max(best, _iou(boxes[i], boxes[j]))
        rows.append({"frame": frame, "max_iou": best})
    return pd.DataFrame(rows)


def summarize_video_features(motion_df: pd.DataFrame, iou_df: pd.DataFrame, context: dict) -> pd.DataFrame:
    """Aggregate per-track/per-frame features into ONE row summarizing the whole
    clip — the exact row shape the classical-ML pipeline (src/models.py,
    src/preprocessing.py) was fit on. Returns a single-row DataFrame with raw
    (unencoded, unscaled) values; column names must match
    src.preprocessing.ALL_FEATURE_COLS exactly.
    """
    row = {
        "road_type": context.get("road_type", "highway"),
        "weather": context.get("weather", "clear"),
        "time_of_day": context.get("time_of_day", "day"),
        "avg_speed": float(motion_df["avg_speed"].mean()) if len(motion_df) else 0.0,
        "max_speed": float(motion_df["max_speed"].max()) if len(motion_df) else 0.0,
        "max_deceleration": float(motion_df["max_deceleration"].min()) if len(motion_df) else 0.0,
        "trajectory_variance": float(motion_df["trajectory_variance"].mean()) if len(motion_df) else 0.0,
        "max_iou": float(iou_df["max_iou"].max()) if len(iou_df) else 0.0,
    }
    return pd.DataFrame([row])
