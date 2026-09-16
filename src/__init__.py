"""RoadSentinel AI — agentic vision pipeline for highway accident/incident detection.

Modules:
    detection        Vehicle detection + tracking (YOLOv8 + ByteTrack)
    features          Motion / overlap feature engineering from tracked boxes
    preprocessing     Leak-safe ColumnTransformer + Pipeline preprocessing
    models            Classical-ML triage classifiers (SVM, RF, LogReg, XGBoost)
    clustering        Severity clustering + spatial (GPS) hotspot clustering
    regression        Emergency response-time regression models
    reasoning_agent   VLM-based severity reasoning agent
    xai               Explainability: SHAP (tabular) + Grad-CAM (vision)
    pipeline          RoadSentinelPipeline — ties every stage together
    dispatch_service  Mock FastAPI + SQLite dispatch/ticketing executor
    precompute        Offline batch job that builds the demo cache used by app.py
"""

__version__ = "1.0.0"
