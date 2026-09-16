# RoadSentinel AI

**Agentic highway accident & incident vision auditor.** Detects and tracks
vehicles in road footage, triages incidents with a leak-safe classical-ML
layer, escalates flagged clips to a VLM reasoning agent for severity
assessment, and auto-drafts a dispatch ticket for high-severity cases —
deployed as a Streamlit app.

The full design rationale, dataset sources, MVP roadmap, and every code
block in this repo are documented in **[`docs/RoadSentinel_AI_Full_Build_Plan.md`](docs/RoadSentinel_AI_Full_Build_Plan.md)** —
read that first if anything here is unclear. This README is the quick-start;
the build plan is the reference.

## Architecture

```
[ Video clip / frames ]
        |
        v
[ 1. Detection & Tracking ]        YOLOv8 (pretrained) + ByteTrack
        |
        v
[ 2. Feature Eng. + Classical-ML Triage ]   SVC / RF / LogReg / XGBoost, K-Means severity clustering
        | flagged only
        v
[ 3. VLM Reasoning Agent ]         severity tier + plain-language explanation
        | if tier >= threshold
        v
[ 4. Mock Dispatch Agent ]         FastAPI + SQLite ticketing
```

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# edit .streamlit/secrets.toml and add your ANTHROPIC_API_KEY

streamlit run app.py
```

The app runs in **Demo-Safe mode by default** (build plan §26): it reads
precomputed results for a curated set of demo clips rather than running the
full pipeline live, which is what makes it safe to demo on Streamlit Cloud's
free tier. To populate that cache:

1. Add your models: train via `notebooks/03_classical_ml.ipynb` and
   `notebooks/04_clustering.ipynb`, or drop pretrained `.joblib` files into
   `models/`.
2. Add a few short demo clips to `demo/` and list them in `app.py`'s
   `SAMPLE_CLIPS` dict.
3. Run `notebooks/08_precompute_demo_cache.ipynb` to populate
   `demo/precomputed/`.

An "Experimental live" mode is also available in the app for actually running
the pipeline against an uploaded clip — it's slower and may lag on free-tier
hosting, which is why it's opt-in rather than the default.

## Repo Layout

```
roadsentinel-ai/
├── app.py                     Streamlit entrypoint (Demo-Safe by default)
├── requirements.txt
├── src/
│   ├── detection.py           YOLOv8 + ByteTrack
│   ├── features.py            Motion/overlap feature engineering
│   ├── preprocessing.py       Leak-safe ColumnTransformer + Pipeline
│   ├── models.py               SVM / RF / LogReg / XGBoost triage classifiers
│   ├── clustering.py           Severity K-Means + spatial hotspot K-Means/PCA/Folium
│   ├── regression.py           Emergency response-time regression
│   ├── imbalance.py            SMOTE / class-weight helpers
│   ├── reasoning_agent.py      VLM severity-reasoning agent (Claude by default)
│   ├── xai.py                  SHAP (tabular) + Grad-CAM (vision) explainability
│   ├── pipeline.py             RoadSentinelPipeline orchestrator
│   ├── dispatch_service.py     Mock FastAPI + SQLite dispatch executor
│   └── precompute.py           Demo-Safe Architecture: builds the demo cache
├── notebooks/                  Training/EDA notebooks, numbered in build order
├── demo/                       Curated demo clips + precomputed cache (committed)
├── models/                     Small trained .joblib artifacts (committed)
├── data/                       Raw datasets (gitignored — see docs §3 for sources)
└── docs/
    └── RoadSentinel_AI_Full_Build_Plan.md   The full plan — architecture, datasets,
                                              every code block, deployment steps, and the
                                              Round-3 fixes (demo-safety + dataset framing)
```

## Datasets

Every dataset this project uses is real, public, and sourced from either
Kaggle or an academic/government release — see **docs/RoadSentinel_AI_Full_Build_Plan.md §3, §21, §22**
for exact links and download commands. In short:

| Task | Dataset | Source |
|---|---|---|
| Detection/tracking | Synthetic accident videos, CADP, CCD | Kaggle + academic release |
| Response-time regression | NYC EMS/911 dispatch data | NYC Open Data |
| Spatial hotspot clustering | US-Accidents | Academic release, mirrored on Kaggle |

This is a **modular platform benchmarked per-capability on the strongest
available public dataset for that task**, not a single end-to-end trained
system — see docs §27 for the exact framing to use in a defense/pitch.

## Deployment

Push to GitHub, then deploy on [Streamlit Community Cloud](https://share.streamlit.io)
pointing at `app.py`. Full steps, including secrets handling and free-tier
gotchas, are in docs §18 and §26.

## License

Add your team/university's preferred license here before making the repo public.
