"""Vehicle detection, tracking, and video annotation for RoadSentinel AI.

Uses pretrained YOLOv8 + ByteTrack for multi-object tracking, and provides
annotated video rendering (bounding boxes, velocity vectors, telemetry HUD).
"""
from __future__ import annotations

import os
import cv2
import numpy as np
import pandas as pd

# COCO vehicle classes: car (2), motorcycle (3), bus (5), truck (7)
VEHICLE_CLASS_IDS = [2, 3, 5, 7]

_model = None


def _get_model():
    """Lazy load YOLO model to optimize startup time."""
    global _model
    if _model is None:
        from ultralytics import YOLO
        _model = YOLO("yolov8n.pt")
    return _model


def extract_tracks(video_path: str, conf: float = 0.35) -> pd.DataFrame:
    """Run YOLOv8 + ByteTrack on a video file.

    Returns a DataFrame with columns: frame, track_id, x, y, w, h
    (coordinates in pixels).
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    model = _get_model()
    results = model.track(
        source=video_path,
        conf=conf,
        classes=VEHICLE_CLASS_IDS,
        tracker="bytetrack.yaml",
        persist=True,
        stream=True,
        verbose=False,
    )

    records = []
    for frame_idx, r in enumerate(results):
        if r.boxes is None or r.boxes.id is None:
            continue
        boxes = r.boxes.xywh.cpu().numpy()
        track_ids = r.boxes.id.cpu().numpy()
        for box, tid in zip(boxes, track_ids):
            x, y, w, h = box
            records.append({
                "frame": frame_idx,
                "track_id": int(tid),
                "x": float(x),
                "y": float(y),
                "w": float(w),
                "h": float(h),
            })

    df = pd.DataFrame(records, columns=["frame", "track_id", "x", "y", "w", "h"])
    if df.empty:
        # Provide defensive fallback so pipeline does not crash on empty/dark video
        return pd.DataFrame(columns=["frame", "track_id", "x", "y", "w", "h"])
    return df


def annotate_video(
    video_path: str,
    output_path: str,
    tracks_df: pd.DataFrame,
    accident_prob: float = 0.0,
    fps: int = 25,
) -> str:
    """Render an annotated MP4 video with track bounding boxes, IDs, and a telemetry HUD."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 360
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Pre-group tracks by frame index for fast lookup
    tracks_by_frame = {}
    if not tracks_df.empty and "frame" in tracks_df.columns:
        for f_idx, g in tracks_df.groupby("frame"):
            tracks_by_frame[int(f_idx)] = g.to_dict("records")

    frame_idx = 0
    is_alert = accident_prob >= 0.50

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Draw vehicle tracking bounding boxes
        frame_tracks = tracks_by_frame.get(frame_idx, [])
        for trk in frame_tracks:
            x, y, w, h = trk["x"], trk["y"], trk["w"], trk["h"]
            tid = trk["track_id"]
            x1, y1 = int(x - w / 2), int(y - h / 2)
            x2, y2 = int(x + w / 2), int(y + h / 2)

            box_color = (0, 0, 230) if is_alert else (230, 140, 20)
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            label = f"TRK #{tid}"
            cv2.putText(frame, label, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 1)

        # Draw Top HUD Banner
        banner_color = (30, 30, 180) if is_alert else (40, 120, 40)
        cv2.rectangle(frame, (0, 0), (width, 35), banner_color, -1)
        status_text = (
            f"ROADSENTINEL AUDIT: [INCIDENT FLAGGED] P(Accident)={accident_prob*100:.1f}%"
            if is_alert
            else f"ROADSENTINEL AUDIT: [SAFE TRAFFIC] P(Accident)={accident_prob*100:.1f}%"
        )
        cv2.putText(frame, status_text, (15, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Rendered annotated video -> {output_path}")
    return output_path


def extract_keyframe(video_path: str, output_frame_path: str) -> str:
    """Extract a representative keyframe (middle frame) from a video."""
    os.makedirs(os.path.dirname(output_frame_path), exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_frame = max(0, total_frames // 2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()

    if ret:
        cv2.imwrite(output_frame_path, frame)
    else:
        # Fallback blank frame if video is unreadable
        blank = np.full((360, 640, 3), 40, dtype=np.uint8)
        cv2.imwrite(output_frame_path, blank)

    cap.release()
    return output_frame_path
