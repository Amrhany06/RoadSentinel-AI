# RoadSentinel AI — Full Build Plan
### Documentation + MVP Roadmap + Setup + Code + Integration + Deployment, all in one place

> **Status: FINAL — Round 3 (post strategic review).** This round closes the three red flags raised in the team's strategic review of Rounds 1–2 — dataset cohesion, Grad-CAM complexity, and Streamlit Cloud live-demo risk — with concrete architecture, not just caveats. See §26–27. Verdict: RoadSentinel AI remains the pick; reasoning is in the accompanying chat response. This document is build-ready.

---

## Table of Contents
1. [Project Summary](#1-project-summary)
2. [Architecture](#2-architecture)
3. [Datasets — Exact Sources & Download Method](#3-datasets--exact-sources--download-method)
4. [Repo Structure](#4-repo-structure)
5. [Environment Setup (Colab + Local)](#5-environment-setup-colab--local)
6. [MVP Roadmap (Phased, Lowest-Risk-First)](#6-mvp-roadmap-phased-lowest-risk-first)
7. [Code — Detection & Tracking](#7-code--detection--tracking)
8. [Code — Feature Engineering](#8-code--feature-engineering)
9. [Code — Preprocessing (Encoding + Scaling)](#9-code--preprocessing-encoding--scaling)
10. [Code — Classical ML (SVM / RF / GridSearch)](#10-code--classical-ml-svm--rf--gridsearch)
11. [Code — Unsupervised Layer (K-Means + Silhouette)](#11-code--unsupervised-layer-k-means--silhouette)
12. [Code — VLM Reasoning Agent](#12-code--vlm-reasoning-agent)
13. [Code — Orchestration Pipeline](#13-code--orchestration-pipeline)
14. [Code — Mock Dispatch Agent (FastAPI + SQLite)](#14-code--mock-dispatch-agent-fastapi--sqlite)
15. [Code — Streamlit App](#15-code--streamlit-app)
16. [Integration — How It All Connects](#16-integration--how-it-all-connects)
17. [GitHub Setup](#17-github-setup)
18. [Streamlit Cloud Deployment](#18-streamlit-cloud-deployment)
19. [Curriculum Coverage Map](#19-curriculum-coverage-map)
20. [Tips, Pitfalls & Judging-Day Checklist](#20-tips-pitfalls--judging-day-checklist)
21. [Gap-Closure: Regression Module — Emergency Response Time](#21-gap-closure-regression-module--emergency-response-time)
22. [Gap-Closure: Spatial Hotspot Clustering (K-Means + PCA + Map)](#22-gap-closure-spatial-hotspot-clustering-k-means--pca--map)
23. [Gap-Closure: Explainable AI — SHAP + Grad-CAM](#23-gap-closure-explainable-ai--shap--grad-cam)
24. [Gap-Closure: Class Imbalance Handling](#24-gap-closure-class-imbalance-handling)
25. [Updated Curriculum Coverage Map (Supersedes §19)](#25-updated-curriculum-coverage-map-supersedes-19)
26. [Demo-Safe Architecture — Precompute-First Design](#26-demo-safe-architecture--precompute-first-design)
27. [Dataset Cohesion — Defense Narrative Script](#27-dataset-cohesion--defense-narrative-script)

---

## 1. Project Summary

**RoadSentinel AI** is an agentic computer-vision system that watches road footage, detects and tracks vehicles, runs a classical-ML triage layer, escalates ambiguous/positive cases to a VLM reasoning agent for severity assessment, and auto-logs a dispatch ticket for high-severity incidents — deployed as a live Streamlit app pulled from GitHub.

---

## 2. Architecture

```
[ Video clip / frames ]
        │
        ▼
┌─────────────────────────────┐
│ 1. Detection & Tracking      │  YOLOv8 (pretrained) + ByteTrack
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│ 2. Feature Engineering +     │  speed, deceleration, IOU overlap,
│    Classical-ML Triage       │  trajectory variance, context (road/
│                               │  weather/time) → SVC / RF / GridSearch
└──────────────┬──────────────┘
               │ flagged only
               ▼
┌─────────────────────────────┐
│ 3. VLM Reasoning Agent        │  severity tier + plain-language
└──────────────┬──────────────┘  explanation
               │ if tier ≥ threshold
               ▼
┌─────────────────────────────┐
│ 4. Mock Dispatch Agent        │  FastAPI + SQLite ticketing
└─────────────────────────────┘
```

Stage 2 also produces a K-Means severity/type cluster (validated by silhouette score) that rides along as extra context into Stage 3.

---

## 3. Datasets — Exact Sources & Download Method

| Dataset | Source Type | Link | Size | Download Method | Role |
|---|---|---|---|---|---|
| **Synthetic Dataset for Accident Detection** | **Kaggle** | `kaggle.com/datasets/mehwishtahir722/synthetic-dataset-for-accident-detection` | Small (synthetic road-accident videos) | `kagglehub.dataset_download(...)` — no auth headaches | Use first — fastest to get an end-to-end pipeline running, and good for oversampling rare severe-accident cases |
| **"Accident Detection From CCTV Footage"** | **Kaggle** | Search Kaggle for this exact title (several forks/re-uploads exist under different usernames — confirm the slug on kaggle.com before running, since dataset ownership occasionally changes) | CCTV frame dataset | `kagglehub.dataset_download("<owner>/<slug>")` once confirmed | Good secondary image-frame source for the MVP v0 static classifier |
| **CADP** (Car Accident Detection & Prediction) | **Academic release** (not Kaggle) | `ankitshah009.github.io/accident_forecasting_traffic_camera` | 1,416 YouTube traffic-camera segments, 205 with full spatio-temporal box annotations | Follow the homepage's download instructions (Google Drive-hosted) | The most "real" CCTV dataset — swap in once your pipeline works, for a credible final training set |
| **CCD** (Car Crash Dataset, built on BDD100K) | **GitHub / academic** | `github.com/Cogito2012/CarCrashDataset` | 1,500 dashcam videos with text descriptions | `git clone`, then follow the repo's README download script | Strong dashcam-angle complement to CADP's CCTV angle |
| **DoTA** (Detection of Traffic Anomaly) | **GitHub / academic** | `github.com/MoonBlvd/Detection-of-Traffic-Anomaly` | 4,677 YouTube dashcam videos — the largest public traffic-anomaly dataset, with when/where/what annotations | `git clone` + `ffmpeg` + a YouTube cookies file to re-download source videos | Stretch-goal dataset for anomaly-type diversity — the cookie/auth step makes it Week-2 material, not Day-1 |

**Recommended sequencing:** start with the two Kaggle sets (zero-friction download, good for MVP v0/v1) → swap in CADP + CCD once the pipeline is proven (these are the datasets you'd cite as "real-world" in your pitch) → treat DoTA as a stretch goal only if time remains, because of its YouTube-cookie download step.

```python
# Kaggle download — works immediately in Colab, no cookies/auth needed
import kagglehub

synthetic_path = kagglehub.dataset_download(
    "mehwishtahir722/synthetic-dataset-for-accident-detection"
)
print("Synthetic accident dataset downloaded to:", synthetic_path)

# Confirm the exact slug for the CCTV footage dataset on kaggle.com first, then:
# cctv_path = kagglehub.dataset_download("<owner>/<slug>")
```

```bash
# CADP — follow homepage instructions (Drive-hosted videos + annotations)
# https://ankitshah009.github.io/accident_forecasting_traffic_camera

# CCD — GitHub-hosted, run its own download script after cloning
git clone https://github.com/Cogito2012/CarCrashDataset.git

# DoTA — stretch goal, needs ffmpeg + YouTube cookies (see repo README)
git clone https://github.com/MoonBlvd/Detection-of-Traffic-Anomaly.git
```

---

## 4. Repo Structure

> **Updated to include the gap-closure modules from §21–24** — three new notebooks, one new `src` file, and the additional model artifacts they produce.

```
roadsentinel-ai/
├── README.md
├── requirements.txt
├── .gitignore
├── .streamlit/
│   └── secrets.toml          # gitignored — API keys live here locally
├── data/                      # gitignored — raw + processed datasets
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_classical_ml.ipynb
│   ├── 04_clustering.ipynb
│   ├── 05_regression_response_time.ipynb     # §21
│   ├── 06_spatial_hotspots.ipynb             # §22
│   └── 07_xai.ipynb                          # §23
├── src/
│   ├── __init__.py
│   ├── detection.py           # YOLO + ByteTrack
│   ├── features.py            # engineer_features, iou_overlap_features
│   ├── preprocessing.py       # ColumnTransformer + Pipeline (leak-safe, §9)
│   ├── models.py              # train/load SVM, RF, XGBoost, K-Means (§10–11, §24)
│   ├── reasoning_agent.py     # VLM call
│   ├── pipeline.py            # RoadSentinelPipeline orchestrator
│   ├── dispatch_service.py    # FastAPI mock dispatch
│   └── xai_vision.py          # Grad-CAM on the secondary CNN classifier (§23)
├── models/                    # gitignored except final small artifacts:
│                               #   svm_triage_pipeline.joblib, severity_kmeans.joblib,
│                               #   response_time_regressor.joblib, hotspot_kmeans.joblib
│                               #   metrics/  (saved ROC/PR/SHAP/silhouette PNGs)
├── demo/                      # curated demo clips + hotspot_map.html for judging day
├── app.py                     # Streamlit entrypoint
└── docs/
    └── RoadSentinel_AI_Full_Build_Plan.md   # this file
```

```
# .gitignore
data/
__pycache__/
*.pyc
.env
.streamlit/secrets.toml
dispatch_tickets.db
*.mp4
*.avi
```

---

## 5. Environment Setup (Colab + Local)

> **`requirements.txt` below is now the single consolidated list** — it folds in every "add to requirements.txt" snippet scattered across §22–24 so you only have to run one install command instead of hunting through the document.

```python
# ── Colab bootstrap cell — run this first in every notebook ──
!pip -q install ultralytics opencv-python-headless scikit-learn kagglehub \
    streamlit fastapi "uvicorn[standard]" python-multipart sqlalchemy \
    anthropic joblib pandas numpy matplotlib seaborn \
    xgboost shap torch torchvision grad-cam imbalanced-learn folium streamlit-folium

from google.colab import drive
drive.mount('/content/drive')

import os
PROJECT_DIR = "/content/drive/MyDrive/RoadSentinelAI"
os.makedirs(PROJECT_DIR, exist_ok=True)
os.chdir(PROJECT_DIR)
print("Working in:", os.getcwd())
```

```
# requirements.txt  (for local dev + Streamlit Cloud) — consolidated, single source of truth
ultralytics>=8.2.0
opencv-python-headless>=4.9.0
numpy
pandas
scikit-learn
matplotlib
seaborn
streamlit
fastapi
uvicorn[standard]
python-multipart
sqlalchemy
kagglehub
anthropic
pillow
joblib
xgboost
shap
torch
torchvision
grad-cam
imbalanced-learn
folium
streamlit-folium
```

Local (non-Colab) setup:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## 6. MVP Roadmap (Phased, Lowest-Risk-First)

> **Extended with a Phase 6** covering the four gap-closure modules — being honest about it: adding regression, spatial clustering, XAI, and imbalance handling is genuinely more work, not a free add-on. If judging day is fixed and time is tight, do Phase 6 in the priority order listed (XAI and the leak-safe pipeline fix are what graders actually check line-by-line; the spatial hotspot map is the most cuttable if you run out of runway).

The principle: **get something end-to-end and deployed on Day 2–3**, then upgrade each stage in place. Never let the team be "almost done" with a pipeline that has never run start-to-finish.

| Phase | Goal | Deliverable | Key Risk Killed |
|---|---|---|---|
| **Phase 0 — Day 0–1** | Scaffold everything | GitHub repo created, `requirements.txt` installs cleanly in Colab, both Kaggle datasets downloaded and previewed | "We can't even open the data" |
| **Phase 1 — MVP v0 (Day 1–3)** | Thinnest possible working demo | Static image classifier only (no tracking yet): pretrained CNN features or simple handcrafted pixel/edge features → SVM baseline on the CCTV frame dataset → bare Streamlit app that takes an uploaded image and shows accident/no-accident + confidence. **Deploy this to Streamlit Cloud immediately**, even though it's simple | "We built something great but never got it deployed" |
| **Phase 2 — MVP v1 (Day 3–5)** | Real video pipeline | Swap in YOLOv8 + ByteTrack detection/tracking on real clips from the synthetic + CADP/CCD sets; engineer speed/IOU/trajectory features; train SVC + RandomForest **inside a leak-safe `Pipeline`** (§9) with `GridSearchCV`; report accuracy/F1/ROC-AUC/PR curve | "The video/motion layer never worked" — and, per the assessment check, "the preprocessing leaks" |
| **Phase 3 — MVP v2 (Day 5–6)** | Unsupervised layer | K-Means clustering of flagged incidents into severity/type buckets, validated with silhouette score across k=2..6 | "No unsupervised-learning coverage" |
| **Phase 4 — MVP v3 (Day 6–7)** | Agentic reasoning | Wire in the VLM reasoning agent, gated behind the triage threshold so only flagged clips are sent (controls API cost) | "The 'agentic' claim has nothing behind it" |
| **Phase 5 — Full System (Day 7–8)** | Executor agent + polish | Mock FastAPI+SQLite dispatch service, `RoadSentinelPipeline` orchestrator tying all stages together, metrics tab in Streamlit, curated demo clips, backup demo video recorded | "Live demo breaks in front of judges" |
| **Phase 6 — Gap-Closure (Day 8–11)** | Close the syllabus checklist | In priority order: (1) XGBoost + PR curves + class-imbalance handling added to §10 (cheap, same notebook), (2) SHAP summary plot on the tabular pipeline (§23), (3) Regression module on the EMS response-time data (§21), (4) secondary CNN classifier + Grad-CAM (§23), (5) spatial hotspot K-Means + PCA + Folium map (§22, most cuttable under time pressure) | "Syllabus coverage looked complete but instructors caught the gaps" |

Each phase's code lives in the sections below in the order you'll build them.

---

## 7. Code — Detection & Tracking

```python
# src/detection.py
from ultralytics import YOLO
import pandas as pd

model = YOLO("yolov8n.pt")  # pretrained on COCO — no training needed for the MVP

VEHICLE_CLASS_IDS = [2, 3, 5, 7]  # car, motorcycle, bus, truck (COCO ids)

def extract_tracks(video_path: str, conf: float = 0.4) -> pd.DataFrame:
    """Run YOLO + ByteTrack on a video, return per-frame per-track box records."""
    results = model.track(
        source=video_path, conf=conf, classes=VEHICLE_CLASS_IDS,
        tracker="bytetrack.yaml", persist=True, stream=True, verbose=False,
    )
    records = []
    for frame_idx, r in enumerate(results):
        if r.boxes is None or r.boxes.id is None:
            continue
        for box, tid in zip(r.boxes.xywh.cpu().numpy(), r.boxes.id.cpu().numpy()):
            x, y, w, h = box
            records.append({"frame": frame_idx, "track_id": int(tid), "x": x, "y": y, "w": w, "h": h})
    return pd.DataFrame(records)
```

---

## 8. Code — Feature Engineering

```python
# src/features.py
import numpy as np
import pandas as pd

def engineer_motion_features(tracks_df: pd.DataFrame, fps: int = 25) -> pd.DataFrame:
    """Per-track motion features from raw box positions over time."""
    feats = []
    for tid, g in tracks_df.groupby("track_id"):
        g = g.sort_values("frame")
        dx, dy = g["x"].diff(), g["y"].diff()
        dt = (g["frame"].diff() / fps).replace(0, np.nan)
        speed = np.sqrt(dx**2 + dy**2) / dt
        accel = speed.diff() / dt
        feats.append({
            "track_id": tid,
            "avg_speed": speed.mean(skipna=True),
            "max_speed": speed.max(skipna=True),
            "max_deceleration": accel.min(skipna=True),
            "trajectory_variance": g[["x", "y"]].var().sum(),
            "duration_frames": len(g),
        })
    return pd.DataFrame(feats).fillna(0)


def _iou(b1, b2) -> float:
    xa, ya = max(b1[0]-b1[2]/2, b2[0]-b2[2]/2), max(b1[1]-b1[3]/2, b2[1]-b2[3]/2)
    xb, yb = min(b1[0]+b1[2]/2, b2[0]+b2[2]/2), min(b1[1]+b1[3]/2, b2[1]+b2[3]/2)
    inter = max(0, xb - xa) * max(0, yb - ya)
    a1, a2 = b1[2]*b1[3], b2[2]*b2[3]
    return inter / (a1 + a2 - inter + 1e-9)


def iou_overlap_features(tracks_df: pd.DataFrame) -> pd.DataFrame:
    """Max pairwise box overlap per frame — a cheap collision proxy."""
    rows = []
    for frame, g in tracks_df.groupby("frame"):
        boxes = g[["x", "y", "w", "h"]].values
        best = 0.0
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                best = max(best, _iou(boxes[i], boxes[j]))
        rows.append({"frame": frame, "max_iou": best})
    return pd.DataFrame(rows)
```

---

## 9. Code — Preprocessing (Encoding + Scaling)

> **Fixed after the syllabus assessment check (see the chat response for the full gap report): the original version of this section fit `LabelEncoder`/`MinMaxScaler` on the whole dataset before splitting — that's textbook data leakage, and the checklist calls it out as a "critical grading point." The version below splits first and fits every transformer only on `X_train`, wrapped in a real `ColumnTransformer` + `Pipeline` so it's impossible to leak by accident later.**

```python
# src/preprocessing.py
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd

CAT_COLS = ["road_type", "weather", "time_of_day"]
NUM_COLS = ["avg_speed", "max_speed", "max_deceleration", "trajectory_variance", "max_iou"]

def split_first(df: pd.DataFrame, label_col: str = "is_accident"):
    """Always split BEFORE fitting anything — this is the leakage-prevention step."""
    X = df[CAT_COLS + NUM_COLS]
    y = df[label_col]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

def build_preprocessor() -> ColumnTransformer:
    """
    OneHotEncoder instead of LabelEncoder for model inputs: LabelEncoder imposes a false
    ordinal relationship (e.g. "fog" > "clear") that hurts SVM/linear models — it's fine
    for encoding a *target* column, not for encoding *feature* columns going into a
    distance- or coefficient-based model.
    """
    return ColumnTransformer(transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
        ("num", MinMaxScaler(), NUM_COLS),
    ])

# usage — fit ONLY on X_train, never on the full dataframe:
# X_train, X_test, y_train, y_test = split_first(df)
# preprocessor = build_preprocessor()
# preprocessor.fit(X_train)                 # fit on train only
# X_train_ready = preprocessor.transform(X_train)
# X_test_ready  = preprocessor.transform(X_test)   # transform only, never fit, on test
```

In practice you won't call `.fit()`/`.transform()` on the preprocessor by hand — Section 10 wraps `build_preprocessor()` directly inside an `sklearn.pipeline.Pipeline` together with the classifier, so `pipeline.fit(X_train, y_train)` fits both the preprocessing and the model on the training split in one call, and `pipeline.predict(X_test)` (or raw new data at inference time) can never see the training-only fit statistics leak into itself.

---

## 10. Code — Classical ML (SVM / RF / XGBoost / GridSearch)

> **Updated after the assessment check: each model is now built as a single `Pipeline(preprocessor + classifier)` (leak-safe, matches §9), XGBoost is added alongside SVM/RF per the syllabus checklist, and evaluation now includes a Precision-Recall curve alongside ROC-AUC** (PR curves matter more than ROC-AUC on an imbalanced accident/no-accident dataset — see §24 for why).

```python
# src/models.py (classical-ML section)
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (classification_report, roc_auc_score, confusion_matrix,
                              PrecisionRecallDisplay, RocCurveDisplay)
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import joblib

from src.preprocessing import build_preprocessor

def train_svm(X_train, y_train):
    pipe = Pipeline([("prep", build_preprocessor()), ("clf", SVC(probability=True))])
    grid = GridSearchCV(
        pipe,
        param_grid={"clf__C": [0.1, 1, 10], "clf__kernel": ["rbf", "linear"], "clf__gamma": ["scale", "auto"]},
        cv=5, scoring="f1", n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    print("Best SVM params:", grid.best_params_)
    return grid.best_estimator_          # a fitted Pipeline: preprocessing + model together

def train_rf(X_train, y_train, class_weight="balanced"):
    pipe = Pipeline([
        ("prep", build_preprocessor()),
        ("clf", RandomForestClassifier(n_estimators=300, class_weight=class_weight, random_state=42)),
    ])
    pipe.fit(X_train, y_train)
    return pipe

def train_logreg(X_train, y_train, class_weight="balanced"):
    """Cheap linear baseline — the checklist explicitly lists Logistic Regression
    alongside SVM/RF/XGBoost, and it's a useful sanity check against the fancier models."""
    from sklearn.linear_model import LogisticRegression
    pipe = Pipeline([
        ("prep", build_preprocessor()),
        ("clf", LogisticRegression(class_weight=class_weight, max_iter=1000)),
    ])
    pipe.fit(X_train, y_train)
    return pipe

def train_xgb(X_train, y_train, scale_pos_weight=None):
    pipe = Pipeline([
        ("prep", build_preprocessor()),
        ("clf", XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,   # set to (neg_count / pos_count) for imbalance — see §24
            eval_metric="logloss", random_state=42,
        )),
    ])
    pipe.fit(X_train, y_train)
    return pipe

def evaluate(name, pipe, X_test, y_test, save_dir="models/metrics"):
    preds = pipe.predict(X_test)
    probs = pipe.predict_proba(X_test)[:, 1]
    print(f"\n--- {name} ---")
    print(classification_report(y_test, preds))
    print("ROC-AUC:", roc_auc_score(y_test, probs))
    print("Confusion matrix:\n", confusion_matrix(y_test, preds))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    RocCurveDisplay.from_predictions(y_test, probs, ax=axes[0])
    axes[0].set_title(f"{name} — ROC Curve")
    PrecisionRecallDisplay.from_predictions(y_test, probs, ax=axes[1])
    axes[1].set_title(f"{name} — Precision-Recall Curve")
    fig.tight_layout()
    fig.savefig(f"{save_dir}/{name.lower()}_curves.png")   # for the Streamlit metrics tab

# usage:
# svm_pipe = train_svm(X_train, y_train)
# rf_pipe  = train_rf(X_train, y_train)
# logreg_pipe = train_logreg(X_train, y_train)
# xgb_pipe = train_xgb(X_train, y_train, scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum())
# evaluate("SVM", svm_pipe, X_test, y_test)
# evaluate("RandomForest", rf_pipe, X_test, y_test)
# evaluate("LogisticRegression", logreg_pipe, X_test, y_test)
# evaluate("XGBoost", xgb_pipe, X_test, y_test)
# joblib.dump(svm_pipe, "models/svm_triage_pipeline.joblib")   # preprocessing + model as ONE artifact
```

---

## 11. Code — Unsupervised Layer (K-Means + Silhouette)

```python
# src/models.py (clustering section)
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib

def find_best_k(features, k_range=range(2, 7)):
    scores = {}
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(features)
        scores[k] = silhouette_score(features, km.labels_)
    best_k = max(scores, key=scores.get)
    print("Silhouette scores by k:", scores, "→ chosen k =", best_k)
    return best_k, scores

def train_kmeans(features, k):
    return KMeans(n_clusters=k, n_init=10, random_state=42).fit(features)

# usage (cluster only the flagged/accident rows):
# flagged = df.loc[df["is_accident"] == 1, NUM_COLS]
# best_k, scores = find_best_k(flagged)
# kmeans = train_kmeans(flagged, best_k)
# joblib.dump(kmeans, "models/severity_kmeans.joblib")
```

---

## 12. Code — VLM Reasoning Agent

```python
# src/reasoning_agent.py
import os, base64, json
import anthropic

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

SEVERITY_PROMPT = """You are a traffic-safety reasoning agent. You are shown a
cropped frame from a flagged video clip, plus computed motion features
(avg_speed, max_deceleration, max_iou, cluster_label). Reason step by step
about what is happening, then respond with STRICT JSON only:
{"severity_tier": <1-5>, "explanation": "<one paragraph>", "recommended_action": "<short>"}"""

def reason_about_clip(frame_path: str, features: dict) -> dict:
    with open(frame_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        system=SEVERITY_PROMPT,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64}},
                {"type": "text", "text": f"Computed features: {features}"},
            ],
        }],
    )
    return json.loads(msg.content[0].text)
```

> Swap `anthropic` for `openai` (GPT-4o) or a local Qwen2.5-VL endpoint the same way — only this file changes; nothing downstream needs to know which VLM you used.

---

## 13. Code — Orchestration Pipeline

> **Updated after the assessment check: `svm_model` is now the single fitted `Pipeline` from §10 (preprocessing baked in), so `_build_feature_row` only needs to build a raw, unencoded, unscaled row — the pipeline's own `ColumnTransformer` handles encoding/scaling identically to how it was fit on training data.** This also removes the separate `scaler`/`encoders` arguments entirely, which is one less place for train/inference mismatch to creep in.

```python
# src/pipeline.py
import pandas as pd
from src.detection import extract_tracks
from src.features import engineer_motion_features, iou_overlap_features

class RoadSentinelPipeline:
    def __init__(self, svm_pipeline, kmeans_model,
                 reasoning_fn, dispatch_fn,
                 triage_threshold: float = 0.5, dispatch_tier: int = 4):
        self.svm_pipeline = svm_pipeline      # fitted sklearn Pipeline: preprocessing + classifier
        self.kmeans_model = kmeans_model
        self.reasoning_fn = reasoning_fn
        self.dispatch_fn = dispatch_fn
        self.triage_threshold = triage_threshold
        self.dispatch_tier = dispatch_tier

    def run(self, video_path: str, context: dict, sample_frame_path: str) -> dict:
        tracks = extract_tracks(video_path)
        motion = engineer_motion_features(tracks)
        iou = iou_overlap_features(tracks)

        row_df = self._build_feature_row(motion, iou, context)      # raw values — no manual encode/scale
        accident_prob = float(self.svm_pipeline.predict_proba(row_df)[0][1])
        result = {"accident_probability": accident_prob, "stage": "triage_only"}

        if accident_prob < self.triage_threshold:
            return result

        num_cols = ["avg_speed", "max_speed", "max_deceleration", "trajectory_variance", "max_iou"]
        cluster = int(self.kmeans_model.predict(row_df[num_cols])[0])
        result["cluster"] = cluster

        reasoning = self.reasoning_fn(sample_frame_path, {"accident_probability": accident_prob, "cluster": cluster})
        result.update(reasoning)
        result["stage"] = "reasoned"

        if result.get("severity_tier", 0) >= self.dispatch_tier:
            ticket = self.dispatch_fn(sample_frame_path, result)
            result["dispatch_ticket"] = ticket
            result["stage"] = "dispatched"

        return result

    def _build_feature_row(self, motion, iou, context) -> pd.DataFrame:
        """Aggregate per-video summary row as a single-row DataFrame with raw
        (unencoded, unscaled) values — column names must match CAT_COLS + NUM_COLS
        from src/preprocessing.py exactly, since the pipeline's ColumnTransformer
        selects columns by name."""
        return pd.DataFrame([{
            "road_type": context["road_type"], "weather": context["weather"], "time_of_day": context["time_of_day"],
            "avg_speed": motion["avg_speed"].mean(), "max_speed": motion["max_speed"].max(),
            "max_deceleration": motion["max_deceleration"].min(),
            "trajectory_variance": motion["trajectory_variance"].mean(),
            "max_iou": iou["max_iou"].max() if len(iou) else 0.0,
        }])
```

---

## 14. Code — Mock Dispatch Agent (FastAPI + SQLite)

```python
# src/dispatch_service.py
from fastapi import FastAPI, Form
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

Base = declarative_base()
engine = create_engine("sqlite:///dispatch_tickets.db")
SessionLocal = sessionmaker(bind=engine)

class Ticket(Base):
    __tablename__ = "tickets"
    id = Column(Integer, primary_key=True)
    severity_tier = Column(Integer)
    explanation = Column(String)
    frame_path = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(engine)
app = FastAPI(title="RoadSentinel Mock Dispatch API")

@app.post("/tickets")
def create_ticket(severity_tier: int = Form(...), explanation: str = Form(...), frame_path: str = Form(...)):
    db = SessionLocal()
    ticket = Ticket(severity_tier=severity_tier, explanation=explanation, frame_path=frame_path)
    db.add(ticket); db.commit(); db.refresh(ticket); db.close()
    return {"ticket_id": ticket.id, "status": "logged"}

@app.get("/tickets")
def list_tickets():
    db = SessionLocal()
    tickets = db.query(Ticket).all()
    db.close()
    return [{"id": t.id, "severity_tier": t.severity_tier, "explanation": t.explanation,
             "created_at": t.created_at.isoformat()} for t in tickets]
```

Run it: `uvicorn src.dispatch_service:app --reload --port 8000`

---

## 15. Code — Streamlit App

> **Updated after the assessment check: loads the single fitted `svm_triage_pipeline.joblib` (preprocessing + model together, from §10) instead of separate scaler/encoder files, and now uses `st.session_state` to keep a running ticket history across reruns** — Session State was called out explicitly in the syllabus checklist and previously wasn't used anywhere in the app.

```python
# app.py
import streamlit as st
import joblib
from src.pipeline import RoadSentinelPipeline
from src.reasoning_agent import reason_about_clip
import requests

st.set_page_config(page_title="RoadSentinel AI", layout="wide")
st.title("🚦 RoadSentinel AI — Agentic Accident & Incident Auditor")

svm_pipeline = joblib.load("models/svm_triage_pipeline.joblib")   # preprocessing + model, one artifact
kmeans_model = joblib.load("models/severity_kmeans.joblib")

if "ticket_history" not in st.session_state:
    st.session_state.ticket_history = []   # persists across reruns within a browser session

def dispatch_fn(frame_path, result):
    resp = requests.post("http://localhost:8000/tickets", data={
        "severity_tier": result["severity_tier"],
        "explanation": result["explanation"],
        "frame_path": frame_path,
    })
    ticket = resp.json()
    st.session_state.ticket_history.append(ticket)
    return ticket

pipeline = RoadSentinelPipeline(svm_pipeline, kmeans_model, reason_about_clip, dispatch_fn)

sample_clips = {
    "Clear collision": "demo/clip_1.mp4",
    "Near-miss": "demo/clip_2.mp4",
    "Normal traffic": "demo/clip_3.mp4",
}

col1, col2 = st.columns(2)
with col1:
    choice = st.selectbox("Try a sample clip (recommended for live demos)", list(sample_clips.keys()))
    uploaded = st.file_uploader("...or upload your own clip", type=["mp4", "mov"])
    video_path = uploaded if uploaded else sample_clips[choice]
    context = {
        "road_type": st.selectbox("Road type", ["highway", "urban", "rural"]),
        "weather": st.selectbox("Weather", ["clear", "rain", "fog", "night"]),
        "time_of_day": st.selectbox("Time of day", ["day", "night"]),
    }

if st.button("Run RoadSentinel"):
    with st.spinner("Detecting, tracking, and reasoning..."):
        result = pipeline.run(video_path, context, sample_frame_path="demo/frame_preview.jpg")
    st.session_state.last_result = result   # session_state also survives the widget-driven rerun below

if "last_result" in st.session_state:
    result = st.session_state.last_result
    with col2:
        st.subheader("Result")
        st.metric("Accident probability", f"{result['accident_probability']*100:.1f}%")
        if "severity_tier" in result:
            st.metric("Severity tier", result["severity_tier"])
            st.info(result["explanation"])
        if "dispatch_ticket" in result:
            st.success(f"Dispatch ticket logged: {result['dispatch_ticket']}")

st.divider()
with st.expander("🎫 Dispatch ticket history (this session)"):
    st.write(st.session_state.ticket_history or "No tickets dispatched yet.")

with st.expander("📊 Model metrics (SVM / RF / XGBoost / K-Means)"):
    st.write("Load and render the PNGs saved by `evaluate()` in §10 (ROC + PR curves) and the "
             "silhouette/PCA plots from §22 — export once in your training notebook, `st.image()` them in here.")

with st.expander("🧠 Why did the model decide this? (SHAP / Grad-CAM)"):
    st.write("Render the SHAP summary plot (§23) for the classical-ML triage decision and the "
             "Grad-CAM heatmap for the YOLO/CNN detection here.")
```

---

## 16. Integration — How It All Connects

```
Streamlit app.py
   │
   ├─ loads models/*.joblib  (trained in notebooks 03 & 04, Sections 10–11)
   │
   ├─ calls src/pipeline.py  RoadSentinelPipeline.run()
   │      │
   │      ├─ src/detection.py     extract_tracks()          [YOLO + ByteTrack]
   │      ├─ src/features.py      engineer_motion_features() + iou_overlap_features()
   │      ├─ (inline)             SVM triage predict_proba()
   │      ├─ (inline)             K-Means predict()  [only if flagged]
   │      ├─ src/reasoning_agent.py  reason_about_clip()     [only if flagged]
   │      └─ dispatch_fn() → POST http://localhost:8000/tickets   [only if severity ≥ threshold]
   │             │
   │             └─ src/dispatch_service.py  FastAPI + SQLite    [must be running separately]
   │
   └─ renders results back to the user
```

Two processes run side by side locally: `uvicorn src.dispatch_service:app` (port 8000) and `streamlit run app.py`. On Streamlit Cloud, either run the dispatch logic as an in-process function call instead of an HTTP call (simplest — just call `create_ticket_directly()` rather than `requests.post`), or deploy the FastAPI service separately (e.g., on Render/Railway) and point `dispatch_fn` at its public URL. For a hackathon demo, **in-process is simpler and has one fewer moving part to fail live**.

---

## 17. GitHub Setup

```bash
# one-time
git init
git remote add origin https://github.com/<your-org>/roadsentinel-ai.git
git add .
git commit -m "Initial scaffold: repo structure, requirements, MVP plan"
git push -u origin main

# team workflow: one feature branch per pipeline stage
git checkout -b feature/detection-tracking
git checkout -b feature/classical-ml
git checkout -b feature/vlm-agent
git checkout -b feature/streamlit-app
# open PRs into main, at least one teammate reviews before merge
```

README.md should include: problem statement, architecture diagram, dataset sources + licenses, setup instructions, and a link to this full build plan under `docs/`.

---

## 18. Streamlit Cloud Deployment

1. Push the repo to GitHub (public, or private with the Streamlit Cloud GitHub App installed).
2. Go to `share.streamlit.io` → **New app** → select the repo/branch → main file path `app.py`.
3. Add secrets under **App → Settings → Secrets** (`.streamlit/secrets.toml` format):
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   ```
4. Model artifacts: `models/*.joblib` for the SVM and K-Means are KB-sized — commit them directly to the repo rather than gitignoring them, so the deployed app has something to load.
5. Keep `ultralytics`/`opencv-python-headless` as CPU-only installs (already the case in `requirements.txt` above) to avoid Streamlit Cloud build timeouts — you don't need GPU for YOLOv8n inference on short clips.
6. After first deploy, test with the **curated demo clips**, not arbitrary uploads, before your judging session.

---

## 19. Curriculum Coverage Map

| Syllabus Topic | Where It Lives |
|---|---|
| Python fundamentals | Entire codebase |
| Data analysis / EDA | `notebooks/01_eda.ipynb` |
| Feature engineering | `src/features.py` |
| Label encoding | `src/preprocessing.py` |
| Min-max scaling | `src/preprocessing.py` |
| Supervised classification | `src/models.py` — SVC, RandomForest |
| SVM/SVC + accuracy/scores | `src/models.py` `evaluate()` |
| Grid search / hyperparameter tuning | `src/models.py` `train_svm()` |
| Unsupervised learning / clustering | `src/models.py` `train_kmeans()` |
| Silhouette score | `src/models.py` `find_best_k()` |
| Regression | Add a secondary continuous "risk score" head using `SVR`/`LinearRegression` on the same feature set — one extra cell in `03_classical_ml.ipynb` |
| ML concepts (train/test split, CV, bias-variance) | `notebooks/03_classical_ml.ipynb` |
| Deep learning concepts | YOLOv8 (CNN detector), VLM reasoning agent (transformer) |
| Agentic workflow | `src/pipeline.py` orchestration + `src/dispatch_service.py` executor |

---

## 20. Tips, Pitfalls & Judging-Day Checklist

- **Deploy early, deploy ugly.** Phase 1's static-image MVP should be live on Streamlit Cloud by Day 3 even if it's the simplest model — a deployed simple thing beats an undeployed impressive thing.
- **Gate the VLM calls** behind the triage threshold — both cheaper and a legitimate "agentic efficiency" talking point.
- **Curate 3–5 bulletproof demo clips** and default the app to them; arbitrary live uploads are the #1 cause of failed live demos.
- **Commit small model artifacts** (`.joblib` files) directly — don't rely on retraining or external downloads at deploy time.
- **Run the dispatch logic in-process for the demo**, not as a second server, unless a teammate is comfortable keeping two processes alive during judging.
- **Address privacy/ethics in your pitch**: all footage is from public research/Kaggle datasets, no real personal data, and note you'd blur plates/faces in a production version.
- **Put the GridSearchCV results and silhouette plot on a slide**, not just buried in a notebook — they're your most visible proof of classical-ML + unsupervised-learning coverage.
- **Record a backup demo video** the night before, in case live inference or Wi-Fi fails during judging.

---

## 21. Gap-Closure: Regression Module — Emergency Response Time

The original plan treated regression as an optional add-on. A syllabus review flagged it as a required, load-bearing module, not an afterthought — and honestly, none of the accident-video datasets in §3 (CADP/CCD/DoTA/Kaggle) contain real response-time labels, so this module needs its own dataset.

| Dataset | Source | Link | Role |
|---|---|---|---|
| **EMS Incident Dispatch Data / "911 End-to-End Data"** | **NYC Open Data** (government, not Kaggle) | `data.cityofnewyork.us` — search NYC Open Data for "EMS Incident Dispatch Data"; the "911 End-to-End Data" table (landing page `data.cityofnewyork.us/d/t7p9-n9dy`) has timestamped response segments in seconds | Trains the general severity/distance/traffic → response-time relationship |

**Be upfront about this in your defense:** the absolute response-time *minutes* are NYC-specific and won't transfer to Egypt as literal numbers — what transfers, and what you're actually demonstrating, is the learned *relationship* between severity/distance/traffic-density and response time. Say this explicitly rather than implying it's Egypt-calibrated; instructors respect disclosed limitations far more than an unexamined claim.

```python
# notebooks/05_regression_response_time.ipynb
import pandas as pd, numpy as np, joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

ems_df = pd.read_csv("data/nyc_ems_response.csv")
# engineer: response_minutes (target), incident_severity_level, borough, time_of_day,
# is_life_threatening, call_volume_density (proxy for traffic/congestion)

REG_CAT = ["borough", "time_of_day", "incident_severity_level"]
REG_NUM = ["call_volume_density"]

X = ems_df[REG_CAT + REG_NUM]
y = ems_df["response_minutes"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

reg_preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), REG_CAT),
    ("num", MinMaxScaler(), REG_NUM),
])

ridge_pipe = Pipeline([("prep", reg_preprocessor), ("reg", Ridge())])
ridge_grid = GridSearchCV(ridge_pipe, param_grid={"reg__alpha": [0.1, 1.0, 10.0]}, cv=5, scoring="r2")
ridge_grid.fit(X_train, y_train)

rf_reg_pipe = Pipeline([("prep", reg_preprocessor), ("reg", RandomForestRegressor(n_estimators=300, random_state=42))])
rf_reg_pipe.fit(X_train, y_train)

def report_regression(name, model, X_test, y_test):
    preds = model.predict(X_test)
    print(f"--- {name} ---")
    print("R2:", r2_score(y_test, preds))
    print("MAE:", mean_absolute_error(y_test, preds))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, preds)))

report_regression("Ridge (tuned)", ridge_grid.best_estimator_, X_test, y_test)
report_regression("RandomForestRegressor", rf_reg_pipe, X_test, y_test)

joblib.dump(rf_reg_pipe, "models/response_time_regressor.joblib")
```

**Wiring it into RoadSentinel:** at inference time, feed the CV pipeline's own `severity_tier` (plus an assumed/looked-up distance-to-nearest-facility and time-of-day) into `response_time_regressor.predict(...)` and show the predicted minutes-to-arrival on the generated dispatch ticket in §14/§15.

---

## 22. Gap-Closure: Spatial Hotspot Clustering (K-Means + PCA + Map)

The original §11 K-Means only clustered *incident type/severity* from motion features — the checklist specifically wants **geographic hotspot mapping**: K-Means on real accident GPS coordinates, visualized with PCA and an interactive map.

> **Per §26, this map is always precomputed and committed as static HTML — never re-clustered on a live request.** The code below runs once in a notebook, not in the deployed app.

| Dataset | Source | Link | Size | Role |
|---|---|---|---|---|
| **US-Accidents** (Sobhan Moosavi et al.) | **Academic release, mirrored on Kaggle** | Homepage: `smoosavi.org/datasets/us_accidents` — also searchable on Kaggle as "US Accidents (2016-2023)" | ~7.7 million accident records with GPS lat/long, severity, and weather | The standard, heavily-cited real dataset for exactly this kind of spatial hotspot analysis |

```python
# notebooks/06_spatial_hotspots.ipynb
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import folium
import joblib

acc_df = pd.read_csv("data/us_accidents.csv", usecols=["Start_Lat", "Start_Lng", "Severity"])
acc_df = acc_df.dropna().sample(50_000, random_state=42)   # sample for a fast Colab run

coords = acc_df[["Start_Lat", "Start_Lng"]]
scaled_coords = StandardScaler().fit_transform(coords)

scores = {}
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(scaled_coords)
    scores[k] = silhouette_score(scaled_coords, km.labels_)
best_k = max(scores, key=scores.get)
print("Silhouette scores by k:", scores, "→ chosen k =", best_k)

kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit(scaled_coords)
acc_df["hotspot_cluster"] = kmeans.labels_

pca = PCA(n_components=2).fit(scaled_coords)
print("PCA explained variance ratio:", pca.explained_variance_ratio_)   # the "elbow"/variance plot the checklist wants

# Interactive Folium map — embed with `streamlit-folium` in app.py
m = folium.Map(location=[acc_df.Start_Lat.mean(), acc_df.Start_Lng.mean()], zoom_start=5)
for _, row in acc_df.sample(2000, random_state=1).iterrows():   # subsample markers for map performance
    folium.CircleMarker([row.Start_Lat, row.Start_Lng], radius=3,
                         color=f"C{int(row.hotspot_cluster)}", fill=True).add_to(m)
m.save("demo/hotspot_map.html")

joblib.dump(kmeans, "models/hotspot_kmeans.joblib")
```

```
# add to requirements.txt
folium
streamlit-folium
```

```python
# app.py addition — embed the interactive hotspot map in Streamlit
from streamlit_folium import st_folium
import folium

with st.expander("🗺️ Historical Accident Hotspot Map (US-Accidents, K-Means + PCA)"):
    m = folium.Map(location=[39.5, -98.35], zoom_start=4)  # rebuild or load saved map object here
    st_folium(m, width=700, height=450)
```

---

## 23. Gap-Closure: Explainable AI — SHAP + Grad-CAM

XAI was entirely absent from the original plan. The checklist calls it out as its own module (#8) and asks for it to be "prominent," not buried.

**SHAP for the tabular triage models (§10):**

```python
# notebooks/07_xai.ipynb
import shap
import joblib
import matplotlib.pyplot as plt

xgb_pipe = joblib.load("models/xgb_triage_pipeline.joblib")
X_sample = X_test.sample(200, random_state=42)   # SHAP is slow on full test sets — sample for the demo

# TreeExplainer is fast and exact for RandomForest/XGBoost; use shap.Explainer(pipe.predict_proba, X_sample)
# generically if you swap in SVM instead
explainer = shap.TreeExplainer(xgb_pipe.named_steps["clf"])
X_transformed = xgb_pipe.named_steps["prep"].transform(X_sample)
shap_values = explainer.shap_values(X_transformed)

shap.summary_plot(shap_values, X_transformed, show=False)
plt.savefig("models/metrics/shap_summary.png", bbox_inches="tight")
```

**Grad-CAM for the vision side — one honest caveat:** Grad-CAM is designed for CNN *classifiers* with a clear final conv layer; applying it cleanly to a YOLO *detection* head is non-trivial and not worth the engineering time for an MVP. The practical fix: alongside YOLOv8 for detection/tracking, train one small secondary CNN classifier (ResNet18, transfer-learned) purely on cropped accident-vs-no-accident frames — this gives you a clean, standard Grad-CAM target without fighting YOLO's architecture. **Per §26, this secondary model only ever runs offline against the curated demo clips — it is never part of the live inference path**, which directly removes the "heavy pipeline" risk a strategic review of this plan correctly flagged.

```python
# src/xai_vision.py
import torch, torchvision
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np, cv2

# small secondary classifier, transfer-learned on cropped accident/no-accident frames
resnet = torchvision.models.resnet18(weights="IMAGENET1K_V1")
resnet.fc = torch.nn.Linear(resnet.fc.in_features, 2)   # fine-tune this head on your cropped-frame dataset
resnet.eval()

target_layer = [resnet.layer4[-1]]
cam = GradCAM(model=resnet, target_layers=target_layer)

def explain_frame(frame_bgr: np.ndarray):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    input_tensor = torchvision.transforms.functional.to_tensor(rgb).unsqueeze(0)
    grayscale_cam = cam(input_tensor=input_tensor)[0]
    overlay = show_cam_on_image(rgb.astype(np.float32) / 255.0, grayscale_cam, use_rgb=True)
    return overlay   # save/display this heatmap next to the original crop
```

```
# add to requirements.txt
shap
torch
torchvision
grad-cam
```

Surface both in the Streamlit "Why did the model decide this?" expander added in §15.

---

## 24. Gap-Closure: Class Imbalance Handling

Untouched in the original plan — and accident-video data is inherently imbalanced (most frames are normal traffic; severe accidents are rare). Two complementary fixes, both already partially wired into §10:

```python
# option A — class_weight (already used for RandomForest/XGBoost in §10, zero extra dependencies)
# RandomForestClassifier(..., class_weight="balanced")
# XGBClassifier(..., scale_pos_weight=(neg_count / pos_count))

# option B — SMOTE oversampling, for when class_weight isn't enough
# requires imbalanced-learn's OWN Pipeline class (regular sklearn.pipeline.Pipeline
# does not support samplers mid-pipeline)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from src.preprocessing import build_preprocessor

smote_svm_pipe = ImbPipeline([
    ("prep", build_preprocessor()),
    ("smote", SMOTE(random_state=42)),   # only ever applied to the TRAINING fold during .fit()
    ("clf", SVC(probability=True)),
])
smote_svm_pipe.fit(X_train, y_train)   # SMOTE never touches X_test — no leakage
```

```
# add to requirements.txt
imbalanced-learn
```

Report the class balance explicitly in `notebooks/01_eda.ipynb` (a simple `y.value_counts(normalize=True)` bar chart) — instructors will ask why you chose `class_weight` vs. SMOTE, and "here's the actual imbalance ratio we measured" is the right answer to have ready.

---

## 25. Updated Curriculum Coverage Map (Supersedes §19)

| Syllabus Module (from the assessment checklist) | Where It Lives Now |
|---|---|
| 1. Python fundamentals & data handling | Entire codebase; `src/` is modular `.py`, not notebook-only |
| 2. Preprocessing & leakage control | §9 — `ColumnTransformer` + `Pipeline`, fit only on `X_train` |
| 3. Classical classification (SVM, RF, XGBoost, LogReg) | §10 — all four, each as its own `Pipeline` function |
| 4. Classical regression | §21 — Ridge + RandomForestRegressor on the EMS response-time data, R²/MAE/RMSE |
| 5. Model optimization & pipelines | §9–10 — `Pipeline`, `ColumnTransformer`, `GridSearchCV` |
| 6. Unsupervised learning & clustering | §11 (incident-type K-Means) + §22 (geographic hotspot K-Means + PCA) |
| 7. Deep learning | YOLOv8 (§7) + secondary ResNet18 classifier (§23) + VLM reasoning agent (§12) |
| 8. Explainable AI | §23 — SHAP for tabular, Grad-CAM for vision |
| 9. Deployment, UI & architecture | §15/§17/§18 — Streamlit + `st.session_state` + GitHub + Streamlit Cloud |

This is the table to put on your final defense slide — one row per syllabus module, one section number each instructor can flip to.

---

## 26. Demo-Safe Architecture — Precompute-First Design

This closes the single most serious risk raised in the team's strategic review: Streamlit Cloud's free tier (1 vCPU / 1GB RAM) cannot reliably run live YOLO tracking, a 50,000-point Folium render, and multi-model inference in front of judges without a real chance of a timeout or frozen page. The fix is architectural, not a tip in a bullet list — **the deployed app defaults to serving precomputed results for a curated set of clips, with true live inference demoted to an explicitly-labeled, optional "experimental" mode.**

What runs precomputed (offline, in Colab, committed to the repo) vs. what's allowed to run live in front of judges:

| Component | Precomputed (default path) | Live (optional, clearly labeled) |
|---|---|---|
| YOLO detection + tracking on the 3–5 curated demo clips | ✅ run once in Colab, save the annotated video + features to `demo/precomputed/` | Only for a short (<10s) user-uploaded clip, labeled "may take up to a minute" |
| Classical-ML triage (SVM/RF/XGBoost) | ✅ precomputed alongside the above, for consistency | Also fine live — this stage is cheap, no real risk |
| VLM reasoning agent | ✅ precompute the explanation text per demo clip (one cached API call each) | A live call is fast enough to be safe on its own |
| Hotspot map (§22) | ✅ **always precomputed** — never re-cluster GPS points on a live request | Not offered live — this was the review's highest-risk item |
| SHAP summary plot (§23) | ✅ always a precomputed PNG | Not offered live — SHAP on SVM is slow enough to be a real risk |
| Grad-CAM heatmaps (§23) | ✅ always precomputed for the demo clips' key frames | Not offered live — the secondary ResNet18 never runs during judging |

```python
# app.py — precompute-first loading pattern
import json, streamlit as st

DEMO_CACHE = "demo/precomputed"

def load_cached_result(clip_name: str) -> dict:
    with open(f"{DEMO_CACHE}/{clip_name}.json") as f:
        return json.load(f)

mode = st.radio("Mode", ["Curated demo (instant, cached)", "Experimental live upload (may be slow)"])

if mode.startswith("Curated"):
    choice = st.selectbox("Pick a demo clip", list(sample_clips.keys()))
    result = load_cached_result(choice)             # instant — zero model calls during judging
    st.video(f"{DEMO_CACHE}/{choice}_annotated.mp4")
else:
    uploaded = st.file_uploader("Upload a short clip (<10s recommended)", type=["mp4", "mov"])
    if uploaded and st.button("Run live (experimental)"):
        with st.spinner("This can take up to a minute on the free tier..."):
            result = pipeline.run(uploaded, context, sample_frame_path="tmp_frame.jpg")
```

Build the cache once, offline:

```python
# notebooks/08_precompute_demo_cache.ipynb
import json, os

os.makedirs("demo/precomputed", exist_ok=True)
for name, path in sample_clips.items():
    result = pipeline.run(path, default_context, sample_frame_path=f"demo/{name}_frame.jpg")
    with open(f"demo/precomputed/{name}.json", "w") as f:
        json.dump(result, f)
    # also save the annotated video, this clip's SHAP PNG, and its Grad-CAM PNG here
```

This one architectural change removes essentially all of the live-demo risk the review flagged, while still letting a curious judge try the experimental live mode on their own terms — with an honest expectation set upfront, not a silent freeze mid-presentation.

---

## 27. Dataset Cohesion — Defense Narrative Script

Addresses the review's dataset-cohesion concern directly. The fix isn't technical — it's framing, and it needs to be said out loud in the pitch rather than left for a judge to raise first. Don't pitch this as "a deployed Egyptian smart-traffic system." Pitch it as what it actually is: **a modular agentic vision-analytics platform for traffic safety, with each capability benchmarked on the strongest available public dataset for that specific task** — standard, accepted practice in applied ML work, not a weakness, as long as you state it upfront instead of implying otherwise.

Suggested slide language:

> "RoadSentinel AI is a modular platform, not a single end-to-end trained model. Each stage is validated on the strongest public dataset available for that specific capability: CADP and a Kaggle synthetic set for detection and tracking, NYC's public EMS dispatch data to learn the general relationship between incident severity and response time, and the US-Accidents dataset to demonstrate geographic hotspot clustering at a scale no single-country dataset currently offers publicly. In a production deployment, each of these would be swapped for the equivalent Egyptian data source — local traffic-camera footage, local ambulance dispatch logs, and local accident-location records — without changing the architecture."

Three things to have ready if pressed further:
1. **Name what you'd swap in for Egypt**, for each dataset, even without access to it — this shows you understand the gap rather than glossing over it.
2. **Point to §21's disclosed caveat** about response-time minutes not literally transferring — having flagged this before being asked is the strongest possible answer to "did your team think about this?"
3. **Don't over-explain the video-tracking datasets** — CADP and CCD are already real CCTV/dashcam accident data, not a proxy for anything; there's nothing to defend there.
