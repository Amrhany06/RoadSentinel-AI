"""Unit tests for RoadSentinel AI core components.
"""
from __future__ import annotations

import os
import sys
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dispatch_service import create_ticket, list_tickets
from src.features import engineer_motion_features, iou_overlap_features, summarize_video_features
from src.preprocessing import ALL_FEATURE_COLS, build_preprocessor, split_first
from src.reasoning_agent import reason_about_clip


class TestRoadSentinelComponents(unittest.TestCase):
    def test_feature_engineering_motion(self):
        tracks = pd.DataFrame([
            {"frame": 0, "track_id": 1, "x": 100, "y": 100, "w": 50, "h": 30},
            {"frame": 1, "track_id": 1, "x": 120, "y": 100, "w": 50, "h": 30},
            {"frame": 2, "track_id": 1, "x": 125, "y": 100, "w": 50, "h": 30},
        ])
        motion = engineer_motion_features(tracks, fps=25)
        self.assertEqual(len(motion), 1)
        self.assertIn("avg_speed", motion.columns)
        self.assertIn("max_deceleration", motion.columns)

    def test_iou_overlap(self):
        tracks = pd.DataFrame([
            {"frame": 0, "track_id": 1, "x": 100, "y": 100, "w": 50, "h": 50},
            {"frame": 0, "track_id": 2, "x": 120, "y": 100, "w": 50, "h": 50},
        ])
        iou_df = iou_overlap_features(tracks)
        self.assertEqual(len(iou_df), 1)
        self.assertGreater(float(iou_df["max_iou"].iloc[0]), 0.0)

    def test_preprocessing_leak_safety(self):
        df = pd.DataFrame({
            "road_type": ["highway", "urban", "rural", "highway", "urban"],
            "weather": ["clear", "rain", "fog", "clear", "night"],
            "time_of_day": ["day", "night", "day", "night", "day"],
            "avg_speed": [50.0, 60.0, 45.0, 55.0, 70.0],
            "max_speed": [70.0, 80.0, 65.0, 75.0, 90.0],
            "max_deceleration": [-20.0, -40.0, -10.0, -30.0, -5.0],
            "trajectory_variance": [10.0, 50.0, 5.0, 25.0, 15.0],
            "max_iou": [0.1, 0.6, 0.0, 0.4, 0.05],
            "is_accident": [0, 1, 0, 1, 0],
        })
        X_train, X_test, y_train, y_test = split_first(df, label_col="is_accident", test_size=0.4)
        self.assertEqual(len(X_train) + len(X_test), len(df))
        preprocessor = build_preprocessor()
        preprocessor.fit(X_train)
        X_train_trans = preprocessor.transform(X_train)
        X_test_trans = preprocessor.transform(X_test)
        self.assertEqual(X_train_trans.shape[0], len(X_train))
        self.assertEqual(X_test_trans.shape[0], len(X_test))

    def test_reasoning_agent_fallback(self):
        features = {
            "accident_probability": 0.88,
            "max_iou": 0.55,
            "max_deceleration": -65.0,
            "cluster": 2,
        }
        frame_path = "demo/clip_1_frame.jpg"
        res = reason_about_clip(frame_path, features)
        self.assertIn("severity_tier", res)
        self.assertIn("explanation", res)
        self.assertGreaterEqual(res["severity_tier"], 4)

    def test_dispatch_cad_service(self):
        ticket = create_ticket(severity_tier=4, explanation="Test high-speed rollover", predicted_eta_minutes=7.2)
        self.assertIn("ticket_id", ticket)
        self.assertEqual(ticket["status"], "DISPATCHED")
        tickets = list_tickets()
        self.assertGreater(len(tickets), 0)


if __name__ == "__main__":
    unittest.main()
