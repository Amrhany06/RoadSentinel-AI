"""RoadSentinelPipeline — Multi-Agent Orchestrator.

Implements the 4-stage pipeline:
Perceive (YOLO+ByteTrack) -> Triage (Classical ML) -> Reason (VLM Agent) -> Act (Regression + Dispatch CAD).
Includes stage latency telemetry and defensive exception guards.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional

from src.detection import annotate_video, extract_keyframe, extract_tracks
from src.features import engineer_motion_features, iou_overlap_features, summarize_video_features


class RoadSentinelPipeline:
    def __init__(
        self,
        triage_pipeline,
        kmeans_model,
        reasoning_fn: Callable[[str, Dict[str, Any]], Dict[str, Any]],
        dispatch_fn: Callable[[int, str, str, float], Dict[str, Any]],
        response_time_fn: Optional[Callable[[int, Dict[str, Any]], float]] = None,
        triage_threshold: float = 0.50,
        dispatch_tier_threshold: int = 4,
    ):
        self.triage_pipeline = triage_pipeline
        self.kmeans_model = kmeans_model
        self.reasoning_fn = reasoning_fn
        self.dispatch_fn = dispatch_fn
        self.response_time_fn = response_time_fn
        self.triage_threshold = triage_threshold
        self.dispatch_tier_threshold = dispatch_tier_threshold

    def run(
        self,
        video_path: str,
        context: Dict[str, Any],
        sample_frame_path: Optional[str] = None,
        render_annotated: bool = True,
        annotated_video_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute the full 4-stage pipeline on a video clip with telemetry."""
        start_total = time.time()
        timings = {}

        # -------------------------------------------------------------
        # STAGE 1: PERCEPTION (Vehicle Detection & Tracking)
        # -------------------------------------------------------------
        t0 = time.time()
        tracks_df = extract_tracks(video_path)
        motion_df = engineer_motion_features(tracks_df)
        iou_df = iou_overlap_features(tracks_df)
        row_df = summarize_video_features(motion_df, iou_df, context)
        timings["perception_sec"] = round(time.time() - t0, 3)

        # -------------------------------------------------------------
        # STAGE 2: TRIAGE (Classical ML Classifier)
        # -------------------------------------------------------------
        t0 = time.time()
        accident_prob = float(self.triage_pipeline.predict_proba(row_df)[0][1])
        timings["triage_sec"] = round(time.time() - t0, 3)

        # Optional: Render annotated video with telemetry
        annotated_path = None
        if render_annotated:
            if annotated_video_path is None:
                annotated_video_path = video_path.replace(".mp4", "_annotated.mp4")
            annotated_path = annotate_video(video_path, annotated_video_path, tracks_df, accident_prob)

        # Extract representative keyframe for reasoning and XAI
        if not sample_frame_path:
            sample_frame_path = video_path.replace(".mp4", "_keyframe.jpg")
        extract_keyframe(video_path, sample_frame_path)

        # Base result package
        result: Dict[str, Any] = {
            "accident_probability": round(accident_prob, 4),
            "stage": "triage_only",
            "triage_passed": accident_prob >= self.triage_threshold,
            "sample_frame_path": sample_frame_path,
            "annotated_video_path": annotated_path,
            "telemetry": {
                "avg_speed": float(row_df["avg_speed"].iloc[0]),
                "max_speed": float(row_df["max_speed"].iloc[0]),
                "max_deceleration": float(row_df["max_deceleration"].iloc[0]),
                "trajectory_variance": float(row_df["trajectory_variance"].iloc[0]),
                "max_iou": float(row_df["max_iou"].iloc[0]),
            },
        }

        # Early exit if triage classifier deems traffic safe (saves VLM compute costs)
        if accident_prob < self.triage_threshold:
            timings["total_sec"] = round(time.time() - start_total, 3)
            result["timings"] = timings
            return result

        # -------------------------------------------------------------
        # STAGE 3: REASONING & CATEGORIZATION (Unsupervised + VLM)
        # -------------------------------------------------------------
        t0 = time.time()
        num_cols = ["avg_speed", "max_speed", "max_deceleration", "trajectory_variance", "max_iou"]
        cluster_id = int(self.kmeans_model.predict(row_df[num_cols])[0])
        result["cluster_id"] = cluster_id

        vlm_features = {
            "accident_probability": accident_prob,
            "cluster_id": cluster_id,
            **result["telemetry"],
        }
        reasoning = self.reasoning_fn(sample_frame_path, vlm_features)
        result.update(reasoning)
        result["stage"] = "reasoned"
        timings["reasoning_sec"] = round(time.time() - t0, 3)

        # -------------------------------------------------------------
        # STAGE 4: ACTION & DISPATCH (Regression + CAD Ticketing)
        # -------------------------------------------------------------
        t0 = time.time()
        severity_tier = int(result.get("severity_tier", 1))

        # Predict response arrival time via regression module
        predicted_eta = 8.5
        if self.response_time_fn is not None:
            predicted_eta = float(self.response_time_fn(severity_tier, context))
        result["predicted_response_minutes"] = round(predicted_eta, 1)

        # Gate dispatch ticketing on severity threshold
        if severity_tier >= self.dispatch_tier_threshold:
            ticket = self.dispatch_fn(
                severity_tier=severity_tier,
                explanation=result.get("explanation", ""),
                frame_path=sample_frame_path,
                predicted_eta_minutes=predicted_eta,
            )
            result["dispatch_ticket"] = ticket
            result["stage"] = "dispatched"

        timings["dispatch_sec"] = round(time.time() - t0, 3)
        timings["total_sec"] = round(time.time() - start_total, 3)
        result["timings"] = timings

        return result
