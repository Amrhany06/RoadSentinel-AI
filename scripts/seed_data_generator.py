"""Seed and synthetic data generator for RoadSentinel AI.

Enables immediate end-to-end development, testing, model training, and demo execution
without waiting for external multi-gigabyte dataset downloads.
"""
import os
import cv2
import numpy as np
import pandas as pd


def generate_tabular_features(n_samples: int = 2500, output_path: str = "data/engineered_features.csv") -> pd.DataFrame:
    """Generate realistic engineered vehicle motion features for accident classification."""
    np.random.seed(42)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Label distribution: ~20% accidents, ~80% normal traffic (realistic imbalance)
    is_accident = np.random.binomial(1, 0.20, n_samples)

    road_types = ["highway", "urban", "rural"]
    weathers = ["clear", "rain", "fog", "night"]
    times = ["day", "night"]

    road_type = np.random.choice(road_types, n_samples, p=[0.5, 0.35, 0.15])
    weather = np.random.choice(weathers, n_samples, p=[0.6, 0.2, 0.1, 0.1])
    time_of_day = np.random.choice(times, n_samples, p=[0.7, 0.3])

    avg_speed = []
    max_speed = []
    max_deceleration = []
    trajectory_variance = []
    max_iou = []

    for acc in is_accident:
        if acc == 1:
            # Accident profile: severe deceleration (negative), high IOU overlap, high trajectory variance
            spd = np.random.uniform(40, 110)
            mspd = spd + np.random.uniform(10, 40)
            decel = np.random.uniform(-95.0, -35.0)  # sharp negative deceleration
            var = np.random.exponential(scale=350.0) + 120.0
            iou = np.random.beta(a=3.0, b=2.0) * 0.85 + 0.15  # 0.15 to 1.0 overlap
        else:
            # Normal traffic / near-miss profile: smooth speeds, mild deceleration, near-zero overlap
            spd = np.random.uniform(30, 95)
            mspd = spd + np.random.uniform(5, 15)
            decel = np.random.uniform(-25.0, -2.0)   # gentle normal braking
            var = np.random.exponential(scale=40.0) + 5.0
            iou = np.random.beta(a=0.5, b=8.0) * 0.08   # near zero overlap

        avg_speed.append(spd)
        max_speed.append(mspd)
        max_deceleration.append(decel)
        trajectory_variance.append(var)
        max_iou.append(iou)

    df = pd.DataFrame({
        "road_type": road_type,
        "weather": weather,
        "time_of_day": time_of_day,
        "avg_speed": avg_speed,
        "max_speed": max_speed,
        "max_deceleration": max_deceleration,
        "trajectory_variance": trajectory_variance,
        "max_iou": max_iou,
        "is_accident": is_accident,
    })

    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} feature records -> {output_path} (Accidents: {is_accident.sum()})")
    return df


def generate_ems_response_data(n_samples: int = 3500, output_path: str = "data/nyc_ems_response.csv") -> pd.DataFrame:
    """Generate realistic EMS response time dataset matching NYC Open Data schema."""
    np.random.seed(42)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    boroughs = ["MANHATTAN", "BROOKLYN", "QUEENS", "BRONX", "STATEN ISLAND"]
    borough = np.random.choice(boroughs, n_samples, p=[0.30, 0.30, 0.20, 0.15, 0.05])
    time_of_day = np.random.choice(["day", "night"], n_samples, p=[0.65, 0.35])
    severity_level = np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.1, 0.25, 0.35, 0.20, 0.10])
    call_density = np.random.uniform(0.1, 1.0, n_samples)

    # Base response time: High severity = priority dispatch (faster response), heavy traffic/call density = delay
    base_minutes = 14.0 - (severity_level * 1.6) + (call_density * 6.5)
    # Borough congestion factor
    borough_delay = {
        "MANHATTAN": 2.5,
        "BROOKLYN": 1.8,
        "BRONX": 1.5,
        "QUEENS": 1.0,
        "STATEN ISLAND": 0.2,
    }
    b_delay = np.array([borough_delay[b] for b in borough])
    noise = np.random.normal(0, 1.2, n_samples)
    response_minutes = np.clip(base_minutes + b_delay + noise, 3.0, 35.0)

    df = pd.DataFrame({
        "borough": borough,
        "time_of_day": time_of_day,
        "incident_severity_level": severity_level,
        "call_volume_density": call_density,
        "response_minutes": response_minutes,
    })

    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} EMS response records -> {output_path}")
    return df


def generate_spatial_hotspots_data(n_samples: int = 5000, output_path: str = "data/us_accidents_sample.csv") -> pd.DataFrame:
    """Generate geospatial accident coordinates clustered around major metropolitan centers."""
    np.random.seed(42)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Metropolitan anchor points (lat, lng, weight)
    centers = [
        (40.7128, -74.0060, 0.30),  # New York
        (34.0522, -118.2437, 0.25), # Los Angeles
        (41.8781, -87.6298, 0.15),  # Chicago
        (29.7604, -95.3698, 0.15),  # Houston
        (37.7749, -122.4194, 0.15), # San Francisco
    ]

    lats = []
    lngs = []
    severities = []

    for lat, lng, weight in centers:
        count = int(n_samples * weight)
        lats.extend(lat + np.random.normal(0, 0.35, count))
        lngs.extend(lng + np.random.normal(0, 0.35, count))
        severities.extend(np.random.choice([1, 2, 3, 4], count, p=[0.1, 0.4, 0.35, 0.15]))

    df = pd.DataFrame({
        "Start_Lat": lats,
        "Start_Lng": lngs,
        "Severity": severities,
    })

    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} geospatial accident coordinates -> {output_path}")
    return df


def generate_synthetic_video_clip(output_path: str, scenario: str = "collision", duration_sec: int = 4, fps: int = 25):
    """Render a synthetic video clip simulating road traffic and vehicle bounding boxes."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    width, height = 640, 360
    total_frames = duration_sec * fps

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    v1_x, v1_y = 60, height // 2 - 25
    v2_x, v2_y = width - 120, height // 2 - 25

    keyframe = None

    for i in range(total_frames):
        # Create road asphalt canvas
        frame = np.full((height, width, 3), 45, dtype=np.uint8)

        # Draw road lane markings
        cv2.line(frame, (0, 100), (width, 100), (255, 255, 255), 3)
        cv2.line(frame, (0, height - 100), (width, height - 100), (255, 255, 255), 3)
        dash_w = 40
        for x in range(0, width, dash_w * 2):
            cv2.line(frame, (x, height // 2), (x + dash_w, height // 2), (0, 215, 255), 2)

        # Vehicle dimensions
        vw, vh = 55, 30

        if scenario == "collision":
            # Two vehicles moving towards center and colliding at frame 50
            if i < 50:
                v1_x += 4.5
                v2_x -= 4.5
            else:
                # Collision impact / rebound
                v1_x += np.random.uniform(-1.5, 1.5)
                v2_x += np.random.uniform(-1.5, 1.5)
                v1_y += np.random.uniform(-2, 2)
                # Draw impact spark / hazard
                cv2.circle(frame, (int(v1_x + vw), int(v1_y + vh//2)), int(15 + (i % 8)), (0, 165, 255), -1)

        elif scenario == "near_miss":
            # Two vehicles approach quickly, vehicle 2 swerves abruptly
            if i < 45:
                v1_x += 3.5
                v2_x -= 4.0
            else:
                v1_x += 2.0
                v2_x -= 3.0
                v2_y = max(115, v2_y - 2.5) # evasive swerve

        else: # "normal"
            # Vehicles cruise in distinct parallel lanes smoothly
            v1_y = height // 2 - 50
            v2_y = height // 2 + 25
            v1_x = (60 + i * 4) % (width + 50)
            v2_x = (width - 100 - i * 3) % (width + 50)

        # Draw Vehicle 1 (Car - Blue)
        cv2.rectangle(frame, (int(v1_x), int(v1_y)), (int(v1_x + vw), int(v1_y + vh)), (220, 100, 30), -1)
        cv2.rectangle(frame, (int(v1_x), int(v1_y)), (int(v1_x + vw), int(v1_y + vh)), (255, 255, 255), 2)
        cv2.putText(frame, "ID:1 [CAR]", (int(v1_x), int(v1_y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Draw Vehicle 2 (Truck / SUV - Crimson)
        cv2.rectangle(frame, (int(v2_x), int(v2_y)), (int(v2_x + vw + 15), int(v2_y + vh + 5)), (50, 50, 210), -1)
        cv2.rectangle(frame, (int(v2_x), int(v2_y)), (int(v2_x + vw + 15), int(v2_y + vh + 5)), (255, 255, 255), 2)
        cv2.putText(frame, "ID:2 [SUV]", (int(v2_x), int(v2_y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # HUD Overlay
        cv2.putText(frame, f"SCENARIO: {scenario.upper()} | FRAME {i}/{total_frames}", (15, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        out.write(frame)

        if i == total_frames // 2:
            keyframe = frame.copy()

    out.release()
    print(f"Generated synthetic video clip -> {output_path}")

    # Also save keyframe for visual reasoning
    keyframe_path = output_path.replace(".mp4", "_frame.jpg")
    if keyframe is not None:
        cv2.imwrite(keyframe_path, keyframe)
        print(f"Saved keyframe -> {keyframe_path}")


def generate_all():
    print("=== Generating Seed Datasets for RoadSentinel AI ===")
    generate_tabular_features()
    generate_ems_response_data()
    generate_spatial_hotspots_data()

    print("\n=== Generating Curated Demo Video Clips ===")
    generate_synthetic_video_clip("demo/clip_1.mp4", scenario="collision")
    generate_synthetic_video_clip("demo/clip_2.mp4", scenario="near_miss")
    generate_synthetic_video_clip("demo/clip_3.mp4", scenario="normal")
    print("=== All Seed Assets Generated Successfully ===")


if __name__ == "__main__":
    generate_all()
