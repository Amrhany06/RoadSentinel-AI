"""Feature engineering: turn raw per-frame tracked bounding boxes into the
motion/overlap features the classical-ML triage layer (src/models.py) uses.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def engineer_motion_features(tracks_df: pd.DataFrame, fps: int = 25) -> pd.DataFrame:
    """Per-track motion features derived from box position over time.

    Returns one row per track_id with: avg_speed, max_speed, max_deceleration,
    trajectory_variance, duration_frames.
    """
    if tracks_df.empty:
        return pd.DataFrame(
            columns=["track_id", "avg_speed", "max_speed", "max_deceleration", "trajectory_variance", "duration_frames"]
        )

    feats = []
    for tid, g in tracks_df.groupby("track_id"):
        g = g.sort_values("frame")
        dx, dy = g["x"].diff(), g["y"].diff()
        dt = (g["frame"].diff() / fps).replace(0, np.nan)
        speed = np.sqrt(dx**2 + dy**2) / dt
        accel = speed.diff() / dt
        feats.append(
            {
                "track_id": tid,
                "avg_speed": speed.mean(skipna=True),
                "max_speed": speed.max(skipna=True),
                "max_deceleration": accel.min(skipna=True),
                "trajectory_variance": g[["x", "y"]].var().sum(),
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
