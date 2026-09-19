# RoadSentinel AI — Real-Time Road Incident Detection & CAD Dispatch System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-ResNet--18-EE4C2C.svg)](https://pytorch.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-0.87_AUC-orange.svg)](https://xgboost.readthedocs.io/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Perception-00FFFF.svg)](https://docs.ultralytics.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Command_Center-FF4B4B.svg)](https://streamlit.io/)

**RoadSentinel AI** is an enterprise-grade autonomous roadway surveillance auditor and emergency Computer-Aided Dispatch (CAD) command center. 

It implements a hierarchical, asymmetric 4-stage pipeline that detects and tracks vehicles in traffic footage, extracts kinematic physics, triages collision incidents using leak-safe classical machine learning, performs deep visual verification using fine-tuned residual networks, assesses multi-tier severity, predicts emergency arrival times, and generates automated CAD dispatch tickets with Explainable AI (XAI) overlays.

---

## 🏛️ System Architecture

```
[ Roadway Surveillance Stream / Video / Images ]
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 1: PERCEPTION & KINEMATICS                            │
│ • YOLOv8n Object Detection + ByteTrack Multi-Object Tracker │
│ • Kinematic Telemetry: Velocity, Deceleration, Overlap (IoU)│
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 2: ASYMMETRIC ML TRIAGE CLASSIFIER                    │
│ • XGBoost Classifier (0.872 ROC-AUC, 83.3% Precision)       │
│ • Benchmarked vs. Random Forest, SVM, Logistic Regression   │
│ • Clears >95% normal traffic early (0 cloud / GPU overhead) │
└──────────────────────────────┬──────────────────────────────┘
                               │ (Incident Flagged: P >= 0.50)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 3: MULTIMODAL VERIFICATION & SEVERITY ASSESSMENT      │
│ • Deep Visual Backbone: ResNet-18 Transfer Learning         │
│   (95.0% Test Accuracy, 100.0% Precision on 990 Images)    │
│ • Explainability: Grad-CAM (layer4[-1]) & SHAP Beeswarm     │
│ • Kinematic Severity Clustering: K-Means (k=2)              │
│ • Strategic Reasoning Agent: Multi-Tier Rubric (Tiers 1–5)  │
│   (Claude/Gemini API support + Deterministic Local Fallback)│
└──────────────────────────────┬──────────────────────────────┘
                               │ (Urgent Dispatch: Tier >= 4)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 4: ACTION, ETA REGRESSION & CAD DISPATCH              │
│ • Response-Time Regressor: Random Forest (NYC EMS data)     │
│ • Automated CAD Ticketing Service (SQLite / REST API)       │
│ • Unit Allocation: Trauma EMS, Heavy Rescue, Highway Patrol │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔬 Model Benchmarks & Scientific Evaluation

### 1. Classical Triage Classifiers (Tabular Kinematic Telemetry)
Evaluated with leak-safe preprocessing (`split-first` rule, `MinMaxScaler` on numeric features, `OneHotEncoder` on road/weather contexts):

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | PR-AUC | Status |
|---|---|---|---|---|---|---|---|
| **XGBoost Classifier** | **78.6%** | **83.3%** | **71.4%** | **0.769** | **0.872** | **0.847** | **Primary Production Model** |
| **Random Forest** | 71.4% | 75.0% | 64.3% | 0.692 | 0.870 | 0.858 | Benchmark Candidate |
| **Support Vector Machine (RBF)** | 71.4% | 80.0% | 57.1% | 0.667 | 0.867 | 0.861 | Benchmark Candidate |
| **Logistic Regression** | 67.9% | 85.7% | 42.9% | 0.571 | 0.867 | 0.869 | Linear Baseline |

### 2. Deep Learning Vision Backbone (PyTorch ResNet-18)
* **Backbone:** ResNet-18 fine-tuned on 990 real-world incident images (`data/accident_images/`).
* **Training Formulation:** Freezes early conv layers; fine-tunes residual `layer4` and replacement classification head (`nn.Linear(512, 2)`).
* **Test Performance (Independent 101-image Test Split):**
  * **Accuracy:** **95.0%**
  * **Precision:** **100.0%** (Zero false alarms on safe traffic)
  * **F1 Score:** **0.944**
* **Weights Artifact:** `models/accident_resnet18.pth` (44.7 MB)

### 3. Emergency Response-Time Regressor
* **Model:** Random Forest Regressor (`n_estimators=250`, `max_depth=8`) trained on NYC EMS / 911 dispatch records.
* **Features:** Borough, time of day, incident severity level (1–5), and historical call volume density.
* **Outputs:** Predicted emergency vehicle arrival ETA in minutes.

### 4. Geospatial Hotspot Intelligence
* **Algorithm:** Unsupervised K-Means clustering ($k=3$) with Principal Component Analysis (PCA) 2D projection on nationwide accident coordinates (`data/us_accidents_sample.csv`).
* **Visualization:** Interactive Folium density heatmap saved to `demo/hotspot_map.html`.

---

## 🔍 Explainable AI (XAI)

RoadSentinel AI embeds dual-modality interpretability to satisfy municipal audit standards:
1. **Visual Attention (Grad-CAM):** Gradient-weighted Class Activation Mapping computes gradients with respect to ResNet-18's final convolutional layer (`layer4[-1]`), generating spatial thermal overlays that highlight crumpled vehicle chassis and shattered windshields.
2. **Feature Attribution (SHAP):** TreeExplainer generates Shapley additive explanations for XGBoost, validating that peak deceleration, bounding-box overlap (IoU), and trajectory deviation govern crash classifications.

---

## 📂 Repository Structure

```
roadsentinel-ai/
├── app.py                     # Single-Page Bento Command Center (Streamlit)
├── requirements.txt           # Environment dependencies
├── dispatch_tickets.db        # SQLite database storing auto-generated CAD tickets
├── src/
│   ├── detection.py           # YOLOv8 + ByteTrack detection & tracking
│   ├── features.py            # Kinematic velocity, acceleration & IoU extraction
│   ├── preprocessing.py       # Leak-safe ColumnTransformer & Pipeline builder
│   ├── models.py              # XGBoost, RF, SVM, LogReg triage training & evaluation
│   ├── clustering.py          # Severity K-Means & Geospatial Hotspot clustering (PCA/Folium)
│   ├── regression.py          # Emergency arrival time regression (Random Forest)
│   ├── reasoning_agent.py     # Multi-tier reasoning agent (Claude/Gemini + Deterministic Fallback)
│   ├── xai.py                 # Grad-CAM (ResNet-18) & SHAP (XGBoost) visualizers
│   ├── pipeline.py            # RoadSentinelPipeline 4-stage orchestrator
│   ├── dispatch_service.py    # CAD ticket creation & SQLite persistence
│   └── precompute.py          # Precomputed telemetry loader for zero-latency demoing
├── scripts/
│   ├── train_models.py        # Master training script for tabular ML, clustering, and regression
│   ├── train_vision_classifier.py  # PyTorch training loop for fine-tuning ResNet-18
│   ├── seed_data_generator.py # Synthetic & benchmark seed data generator
│   └── precompute_all.py      # Telemetry precomputation for demo video streams
├── models/
│   ├── accident_resnet18.pth  # Trained PyTorch ResNet-18 weights (44.7 MB)
│   ├── xgb_triage_pipeline.joblib   # Fitted XGBoost triage pipeline
│   ├── rf_triage_pipeline.joblib    # Fitted Random Forest triage pipeline
│   ├── svm_triage_pipeline.joblib   # Fitted SVM triage pipeline
│   ├── severity_kmeans.joblib       # Fitted K-Means severity clusterer
│   ├── response_time_regressor.joblib # Fitted EMS arrival time regressor
│   ├── hotspot_kmeans.joblib        # Fitted geospatial clustering model
│   └── metrics/               # Evaluation plots (ROC/PR curves, confusion matrices, SHAP)
├── data/                      # Dataset directories (accident images, NYC EMS, US accidents)
├── demo/                      # Video test clips, keyframes, and precomputed telemetry
└── docs/                      # Technical presentation slides, study guides, and full build plan
```

---

## 🚀 Quick Start

### 1. Installation & Environment Setup
```bash
# Clone the repository
git clone https://github.com/Amrhany06/RoadSentinel-AI.git
cd RoadSentinel-AI

# Create virtual environment
python -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch the Streamlit Command Center
```bash
streamlit run app.py
```

### 3. Model Training & Re-evaluation
To retrain the complete suite of ML and DL models from scratch:
```bash
# Retrain classical ML, K-Means clustering, and response-time regression:
python scripts/train_models.py

# Retrain the PyTorch ResNet-18 deep vision backbone:
python scripts/train_vision_classifier.py
```

---

## 📊 Datasets & Sourcing

Every dataset used in this system is real, verifiable, and benchmarked per capability:
* **Roadway Collision Images (990 photos):** Sourced from public roadway accident datasets (`data/accident_images/`) for ResNet-18 transfer learning and Grad-CAM evaluation.
* **Surveillance Crash Video Clips:** Curated real-world collision, near-miss, and continuous traffic footage (`demo/`) for YOLOv8 and ByteTrack tracking.
* **Emergency Dispatch Records:** Sourced from **NYC Open Data** (NYC EMS dispatch telemetry) for response time regression.
* **Geospatial Accident Coordinates:** Sourced from the **US-Accidents** national database for hotspot density clustering.

---

## 👥 Authors & Acknowledgments

* **Project:** RoadSentinel AI — Real-Time Road Incident Detection & CAD Dispatch System
* **Graduation Project:** NTI AI Professional Track (2026)
* **Team Lead:** Amr Hany (Team of 4)
