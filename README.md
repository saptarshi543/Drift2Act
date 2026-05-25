# Drift2Act

## Explainable, Proactive Concept Drift Detection for Clinical Sepsis Prediction Models

---

## Overview

Drift2Act is a three-layer framework that **detects**, **explains**, and **responds to** concept drift in clinical sepsis prediction models. The core of the approach is the **SHAP Attribution Drift Index (SADI)** — a novel feature-level score computed over attribution distributions that detects when a model's *reasoning* changes, not just when inputs shift.

### Key Contributions

1. **SADI (SHAP Attribution Drift Index)** — A novel metric operating on SHAP attribution distributions rather than raw features
2. **Drift2Act Controller** — Three-layer proactive controller: Sensing → Risk Certificate → Cost-Aware Intervention
3. **Fairness-Aware Drift Monitoring** — Tracks demographic fairness (DPD, EOD) per prediction window

## Architecture

```
ICU Data Stream → Preprocessing → Baseline Model (XGBoost)
                                       │
                                 SHAP Explainer
                                       │
                              ┌────────┴────────┐
                              │                 │
                         SADI Engine     Baseline Detectors
                              │          (KS/PSI/ADWIN)
                              └────────┬────────┘
                                       │
                              DRIFT2ACT CONTROLLER
                              ├─ Layer 1: Sensing
                              ├─ Layer 2: Risk Certificate
                              └─ Layer 3: Intervention
                                       │
                              ┌────────┴────────┐
                              │                 │
                        Fairness          Alert Generator
                        Monitor
                              │
                        Streamlit Dashboard
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Add Data

Place PhysioNet 2019 Challenge `.psv` files in `data/raw/training/`.  
If unavailable, the pipeline will generate a synthetic fallback dataset automatically.

### 3. Run the Pipeline

```bash
python scripts/run_pipeline.py
```

This runs the complete pipeline (~10-30 min depending on data size):
- Data loading, preprocessing, drift injection
- XGBoost model training + calibration
- SHAP fingerprinting
- Streaming SADI evaluation
- Baseline detector comparison
- Drift2Act controller evaluation
- Ablation study + statistical significance tests
- Paper figure generation

### 4. Launch Dashboard

```bash
streamlit run dashboard/app.py
```

## Project Structure

```
drift2act/
├── data/
│   ├── raw/training/           # PhysioNet .psv files
│   └── processed/              # Preprocessed outputs
├── src/
│   ├── preprocessing/          # loader, imputer, windower
│   ├── models/                 # XGBoost baseline, calibration
│   ├── shap_monitor/           # SHAP fingerprint, SADI metric
│   ├── drift_detection/        # KS, PSI, ADWIN, adversarial
│   ├── controller/             # Drift2Act 3-layer controller
│   ├── fairness/               # DPD, EOD monitoring
│   └── alerts/                 # Clinical alert generator
├── dashboard/app.py            # Streamlit dashboard
├── scripts/run_pipeline.py     # End-to-end orchestration
├── results/                    # CSV, JSON, LaTeX tables
├── paper/figures/              # Publication-quality PNGs
├── models/                     # Saved model artifacts
├── mlruns/                     # MLflow tracking
├── requirements.txt
└── README.md
```

## Novel Metric — SADI

```
SADI(f, t) = α · KL(S_{t-1}(f) ‖ S_t(f))
           + β · |rank_t(f) − rank_{t-1}(f)| / N
           + γ · 𝟙[sign(μ_{t-1}) ≠ sign(μ_t)]
```

| Component | What It Measures |
|-----------|------------------|
| KL Divergence | Shape shift in SHAP attribution distribution |
| Rank Shift | Feature importance rank change |
| Direction Flip | Protective ↔ harmful attribution reversal |

**Default weights:** α=0.5, β=0.3, γ=0.2

## Drift2Act Controller

| Level | Action | Trigger |
|-------|--------|---------|
| 0 | No Action | Normal operation |
| 1 | Alert | Mild drift detected |
| 2 | Recalibrate | Moderate drift — adjust decision threshold |
| 3 | Partial Retrain | High drift — retrain on recent window |
| 4 | Full Retrain | Severe drift — complete model rebuild |

## Results

After running the pipeline, check:
- `results/drift2act_results.csv` — Per-window metrics
- `results/tables/` — LaTeX-ready comparison tables
- `results/baseline_metrics.json` — Model performance
- `results/drift_ground_truth.json` — Injected drift log
- `paper/figures/` — Publication-quality figures
- `mlruns/` — MLflow experiment tracking

## Tech Stack

| Layer | Library |
|-------|---------|
| ML Model | XGBoost, scikit-learn |
| Explainability | SHAP (TreeExplainer) |
| Drift Detection | Custom SADI, scipy, frouros, river |
| Fairness | fairlearn |
| Dashboard | Streamlit, Plotly |
| Tracking | MLflow |


