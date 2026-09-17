"""Vehicle detection, tracking, and video annotation for RoadSentinel AI.

Uses pretrained YOLOv8 + ByteTrack for multi-object tracking, and provides
annotated video rendering (bounding boxes, velocity vectors, telemetry HUD).
"""
from __future__ import annotations

import os
import cv2
import numpy as np
import pandas as pd

# Roadway classes: person (0), bicycle (1), car (2), motorcycle (3), bus (5), truck (7)
# Person & bicycle are essential to track motorcyclists and fallen riders during collisions
VEHICLE_CLASS_IDS = [0, 1, 2, 3, 5, 7]

_model = None


def _get_model():
    """Lazy load YOLO model to optimize startup time."""
    global _model
    if _model is None:
        from ultralytics import YOLO
        _model = YOLO("yolov8n.pt")
    return _model


def _hungarian_centroid_tracking(results, vid_stride: int = 1) -> list[dict]:
    """Pure Python/SciPy Hungarian centroid tracker.
    
    Acts as a zero-dependency fallback whenever ByteTrack or 'lap'/'lapx'
    is unavailable in the environment (e.g. Streamlit Community Cloud).
    """
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        linear_sum_assignment = None

    active_tracks = {}  # track_id -> {'x': x, 'y': y, 'w': w, 'h': h, 'last_frame': f}
    next_id = 1
    max_dist = 140.0  # max centroid distance in pixels to match
    max_lost = 8 * vid_stride

    records = []
    for f_idx, r in enumerate(results):
        actual_frame = f_idx * vid_stride
        if r.boxes is None or len(r.boxes) == 0:
            continue
        boxes = r.boxes.xywh.cpu().numpy()

        if len(active_tracks) == 0:
            for b in boxes:
                active_tracks[next_id] = {"x": float(b[0]), "y": float(b[1]), "w": float(b[2]), "h": float(b[3]), "last_frame": actual_frame}
                records.append({"frame": actual_frame, "track_id": next_id, "x": float(b[0]), "y": float(b[1]), "w": float(b[2]), "h": float(b[3])})
                next_id += 1
        elif linear_sum_assignment is not None:
            t_ids = list(active_tracks.keys())
            t_coords = np.array([[active_tracks[tid]["x"], active_tracks[tid]["y"]] for tid in t_ids])
            d_coords = boxes[:, :2]

            dist_matrix = np.linalg.norm(t_coords[:, None, :] - d_coords[None, :, :], axis=2)
            row_ind, col_ind = linear_sum_assignment(dist_matrix)

            matched_tracks = set()
            matched_detections = set()
            for r_i, c_i in zip(row_ind, col_ind):
                if dist_matrix[r_i, c_i] < max_dist:
                    tid = t_ids[r_i]
                    b = boxes[c_i]
                    active_tracks[tid] = {"x": float(b[0]), "y": float(b[1]), "w": float(b[2]), "h": float(b[3]), "last_frame": actual_frame}
                    records.append({"frame": actual_frame, "track_id": tid, "x": float(b[0]), "y": float(b[1]), "w": float(b[2]), "h": float(b[3])})
                    matched_tracks.add(tid)
                    matched_detections.add(c_i)

            for c_i, b in enumerate(boxes):
                if c_i not in matched_detections:
                    active_tracks[next_id] = {"x": float(b[0]), "y": float(b[1]), "w": float(b[2]), "h": float(b[3]), "last_frame": actual_frame}
                    records.append({"frame": actual_frame, "track_id": next_id, "x": float(b[0]), "y": float(b[1]), "w": float(b[2]), "h": float(b[3])})
                    next_id += 1

            dead_ids = [tid for tid, data in active_tracks.items() if actual_frame - data["last_frame"] > max_lost]
            for tid in dead_ids:
                del active_tracks[tid]
        else:
            # Simple greedy fallback if scipy is also missing
            for b in boxes:
                records.append({"frame": actual_frame, "track_id": next_id, "x": float(b[0]), "y": float(b[1]), "w": float(b[2]), "h": float(b[3])})
                next_id += 1

    return records


def extract_tracks(video_path: str, conf: float = 0.25, vid_stride: int = 2) -> pd.DataFrame:
    """Run multi-object tracking on a video with automatic fallback for cloud deployments.

    Tries YOLOv8 + ByteTrack first. If 'lap' or C-extensions are missing in the
    environment (e.g. Streamlit Cloud), seamlessly falls back to pure-Python
    Hungarian centroid matching so the application never crashes.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    model = _get_model()
    records = []

    # Attempt primary ByteTrack
    try:
        results = model.track(
            source=video_path,
            conf=conf,
            classes=VEHICLE_CLASS_IDS,
            tracker="bytetrack.yaml",
            persist=True,
            imgsz=480,
            vid_stride=vid_stride,
            stream=True,
            verbose=False,
        )

        for frame_idx, r in enumerate(results):
            actual_frame = frame_idx * vid_stride
            if r.boxes is None or r.boxes.id is None:
                continue
            boxes = r.boxes.xywh.cpu().numpy()
            track_ids = r.boxes.id.cpu().numpy()
            for box, tid in zip(boxes, track_ids):
                x, y, w, h = box
                records.append({
                    "frame": actual_frame,
                    "track_id": int(tid),
                    "x": float(x),
                    "y": float(y),
                    "w": float(w),
                    "h": float(h),
                })
    except Exception as e:
        # Fallback to pure PyTorch prediction + Hungarian assignment
        print(f"[Tracker Notice] ByteTrack unavailable ({type(e).__name__}: {e}), using robust Hungarian fallback.")
        results = model.predict(
            source=video_path,
            conf=conf,
            classes=VEHICLE_CLASS_IDS,
            imgsz=480,
            vid_stride=vid_stride,
            stream=True,
            verbose=False,
        )
        records = _hungarian_centroid_tracking(results, vid_stride=vid_stride)

    df = pd.DataFrame(records, columns=["frame", "track_id", "x", "y", "w", "h"])
    if df.empty:
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

    # Re-encode to H.264 for native browser playback
    try:
        import subprocess
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        h264_temp = output_path.replace(".mp4", "_browser.mp4")
        cmd = [
            ffmpeg_exe,
            "-y",
            "-i", output_path,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "ultrafast",
            "-crf", "23",
            h264_temp,
        ]
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if res.returncode == 0 and os.path.exists(h264_temp) and os.path.getsize(h264_temp) > 0:
            os.replace(h264_temp, output_path)
    except Exception:
        pass

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
