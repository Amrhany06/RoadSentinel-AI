"""Standalone runner for offline demo cache precomputation.

Runs RoadSentinelPipeline on all curated demo clips:
- "Clear collision"
- "Near-miss"
- "Normal traffic"

Outputs JSON results and annotated MP4 clips into demo/precomputed/.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib

from scripts.seed_data_generator import generate_all
from src.dispatch_service import create_ticket
from src.pipeline import RoadSentinelPipeline
from src.precompute import precompute_demo_cache
from src.reasoning_agent import reason_about_clip


def run_precompute():
    print("=" * 60)
    print("       ROADSENTINEL AI — OFFLINE DEMO CACHE RUNNER       ")
    print("=" * 60)

    # Ensure demo clips exist
    if not (os.path.exists("demo/clip_1.mp4") and os.path.exists("demo/clip_2.mp4") and os.path.exists("demo/clip_3.mp4")):
        print("Demo clips missing. Generating synthetic clips...")
        generate_all()

    # Load trained models
    svm_path = "models/svm_triage_pipeline.joblib"
    kmeans_path = "models/severity_kmeans.joblib"
    reg_path = "models/response_time_regressor.joblib"

    if not (os.path.exists(svm_path) and os.path.exists(kmeans_path)):
        print("Trained models not found. Running training pipeline first...")
        from scripts.train_models import run_training_pipeline
        run_training_pipeline()

    svm_pipeline = joblib.load(svm_path)
    kmeans_model = joblib.load(kmeans_path)

    # Optional regressor
    regressor = joblib.load(reg_path) if os.path.exists(reg_path) else None

    def response_time_fn(severity_tier: int, context: dict) -> float:
        if regressor is None:
            return 8.5
        import pandas as pd
        sample_df = pd.DataFrame([{
            "borough": "MANHATTAN",
            "time_of_day": context.get("time_of_day", "day"),
            "incident_severity_level": severity_tier,
            "call_volume_density": 0.5,
        }])
        return float(regressor.predict(sample_df)[0])

    def dispatch_fn(severity_tier: int, explanation: str, frame_path: str, predicted_eta_minutes: float):
        return create_ticket(
            severity_tier=severity_tier,
            explanation=explanation,
            frame_path=frame_path,
            predicted_eta_minutes=predicted_eta_minutes,
        )

    pipeline = RoadSentinelPipeline(
        triage_pipeline=svm_pipeline,
        kmeans_model=kmeans_model,
        reasoning_fn=reason_about_clip,
        dispatch_fn=dispatch_fn,
        response_time_fn=response_time_fn,
        triage_threshold=0.50,
        dispatch_tier_threshold=4,
    )

    sample_clips = {
        "Clear collision": "demo/clip_1.mp4",
        "Near-miss": "demo/clip_2.mp4",
        "Normal traffic": "demo/clip_3.mp4",
    }
    default_context = {"road_type": "highway", "weather": "clear", "time_of_day": "day"}

    precompute_demo_cache(pipeline, sample_clips, default_context, cache_dir="demo/precomputed")
    print("\n[Done] Offline demo cache is populated and ready for instant judging presentation.")


if __name__ == "__main__":
    run_precompute()
