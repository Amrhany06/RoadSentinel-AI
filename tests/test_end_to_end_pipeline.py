"""Integration test for RoadSentinelPipeline end-to-end execution.
"""
from __future__ import annotations

import os
import sys
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from src.preprocessing import build_preprocessor
from src.pipeline import RoadSentinelPipeline
from src.dispatch_service import create_ticket
from src.reasoning_agent import reason_about_clip
from scripts.seed_data_generator import generate_synthetic_video_clip


from sklearn.base import BaseEstimator, ClassifierMixin

class MockClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.classes_ = np.array([0, 1])

    def fit(self, X, y=None):
        return self

    def predict_proba(self, X):
        return np.array([[0.15, 0.85]])



class TestEndToEndPipeline(unittest.TestCase):
    def setUp(self):
        # Generate short test clip
        self.test_clip = "demo/test_clip.mp4"
        generate_synthetic_video_clip(self.test_clip, scenario="collision", duration_sec=2, fps=15)

        # Build mock models
        mock_pipe = Pipeline([
            ("prep", build_preprocessor()),
            ("clf", MockClassifier()),
        ])
        # Fit dummy preprocessor
        dummy_df = pd.DataFrame([{
            "road_type": "highway", "weather": "clear", "time_of_day": "day",
            "avg_speed": 50.0, "max_speed": 60.0, "max_deceleration": -40.0,
            "trajectory_variance": 50.0, "max_iou": 0.5,
        }])
        mock_pipe.named_steps["prep"].fit(dummy_df)

        kmeans = KMeans(n_clusters=2, random_state=42).fit(
            np.array([[50, 60, -40, 50, 0.5], [30, 40, -10, 10, 0.05]])
        )

        self.pipeline = RoadSentinelPipeline(
            triage_pipeline=mock_pipe,
            kmeans_model=kmeans,
            reasoning_fn=reason_about_clip,
            dispatch_fn=lambda severity_tier, explanation, frame_path, predicted_eta_minutes: create_ticket(
                severity_tier=severity_tier,
                explanation=explanation,
                frame_path=frame_path,
                predicted_eta_minutes=predicted_eta_minutes,
            ),
            response_time_fn=lambda tier, ctx: 6.5,
            triage_threshold=0.50,
            dispatch_tier_threshold=4,
        )

    def tearDown(self):
        if os.path.exists(self.test_clip):
            os.remove(self.test_clip)
        kf = self.test_clip.replace(".mp4", "_frame.jpg")
        if os.path.exists(kf):
            os.remove(kf)

    def test_pipeline_execution(self):
        context = {"road_type": "highway", "weather": "clear", "time_of_day": "day"}
        res = self.pipeline.run(self.test_clip, context)

        self.assertIn("accident_probability", res)
        self.assertIn("stage", res)
        self.assertIn("timings", res)
        self.assertEqual(res["stage"], "dispatched")
        self.assertIn("dispatch_ticket", res)
        self.assertIn("severity_tier", res)


if __name__ == "__main__":
    unittest.main()
