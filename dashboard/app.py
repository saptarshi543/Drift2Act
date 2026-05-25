"""
Drift2Act — Streamlit Dashboard
================================

Two operational modes:
  1. Upload & Analyze — Upload a CSV of patient records, get a drift report
  2. Review Pipeline Results — Browse full pipeline outputs with detailed tabs

Designed for hospital ML teams and clinical data scientists who need to
know whether their deployed sepsis model is still reliable.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import pickle
import joblib
import sys
from pathlib import Path
from datetime import datetime

# ── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Drift2Act — Clinical Drift Monitor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        padding: 1.8rem 2.2rem; border-radius: 14px; margin-bottom: 1.5rem;
        color: white; box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .main-header h1 { color: white; margin: 0; font-size: 2.2rem; font-weight: 700; letter-spacing: -0.5px; }
    .main-header p  { color: #b0b0e0; margin: 0.4rem 0 0 0; font-size: 0.95rem; font-weight: 300; }

    .status-banner {
        padding: 1.6rem 2rem; border-radius: 14px; margin: 1rem 0 1.5rem 0;
        text-align: center; color: white; box-shadow: 0 6px 24px rgba(0,0,0,0.25);
    }
    .status-nominal  { background: linear-gradient(135deg, #0d6e3e, #1abc6b); }
    .status-moderate { background: linear-gradient(135deg, #b8860b, #e6a817); color: #1a1a2e; }
    .status-high     { background: linear-gradient(135deg, #c0392b, #e74c3c); }
    .status-critical { background: linear-gradient(135deg, #7b0000, #c0392b); }
    .status-banner h2 { margin: 0 0 0.3rem 0; font-size: 1.8rem; }
    .status-banner p  { margin: 0; font-size: 1.05rem; opacity: 0.92; }

    .kpi-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; }
    .kpi-card {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        border: 1px solid rgba(255,255,255,0.06); border-radius: 12px;
        padding: 1.1rem 1.4rem; text-align: center; color: white; flex: 1;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .kpi-card .kpi-value { font-size: 1.9rem; font-weight: 700; margin: 0.2rem 0; line-height: 1.2; }
    .kpi-card .kpi-label { font-size: 0.72rem; color: #8888aa; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 500; }
    .kpi-card .kpi-delta { font-size: 0.78rem; margin-top: 0.2rem; font-weight: 500; }

    .rec-box {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        border-left: 4px solid #3498db; border-radius: 10px;
        padding: 1.2rem 1.5rem; margin: 1rem 0; color: #ddd;
    }
    .rec-box h4 { margin: 0 0 0.5rem 0; color: white; }

    .alert-box {
        background: #0d1117; border-left: 4px solid #e74c3c; padding: 1rem 1.2rem;
        border-radius: 6px; font-family: 'JetBrains Mono', 'Courier New', monospace;
        font-size: 0.82rem; white-space: pre-wrap; color: #c9d1d9; max-height: 400px;
        overflow-y: auto; line-height: 1.5;
    }

    .upload-zone {
        border: 2px dashed rgba(52,152,219,0.4); border-radius: 14px;
        padding: 2rem; text-align: center; margin: 1rem 0;
        background: rgba(26,26,46,0.5);
    }
    .upload-zone h3 { color: #3498db; margin: 0 0 0.5rem 0; }
    .upload-zone p  { color: #888; font-size: 0.9rem; }

    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        border: 1px solid rgba(255,255,255,0.06); border-radius: 12px;
        padding: 0.9rem 1rem; box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    div[data-testid="stMetric"] label {
        font-size: 0.75rem !important; text-transform: uppercase; letter-spacing: 1px;
    }

    .stTabs [data-baseweb="tab-list"] { gap: 6px; background: rgba(26,26,46,0.5); border-radius: 10px; padding: 4px; }
    .stTabs [data-baseweb="tab"] { border-radius: 8px; padding: 0.5rem 1.1rem; font-weight: 500; font-size: 0.88rem; }

    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0d1117; }
    ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# SHARED UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

VITAL_FEATURES = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2']
LAB_FEATURES = ['BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
                'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Glucose',
                'Lactate', 'Magnesium', 'Phosphate', 'Potassium', 'Hct', 'Hgb',
                'PTT', 'WBC', 'Fibrinogen', 'Platelets', 'TroponinI', 'Bilirubin_total']

PHASE_COLORS = {'phase1': 'rgba(46,204,113,0.12)', 'phase2': 'rgba(241,196,15,0.12)', 'phase3': 'rgba(231,76,60,0.12)'}
PHASE_LINE_COLORS = {'phase1': '#2ecc71', 'phase2': '#f1c40f', 'phase3': '#e74c3c'}


def add_phase_shading(fig, df, row=1, col=1):
    for phase, color in PHASE_COLORS.items():
        pdata = df[df['true_phase'] == phase]
        if len(pdata) > 0:
            fig.add_vrect(
                x0=pdata['window_num'].min() - 0.5, x1=pdata['window_num'].max() + 0.5,
                fillcolor=color, layer="below", line_width=0,
                annotation_text=phase.replace('phase', 'P'),
                annotation_position="top left",
                annotation_font=dict(size=10, color=PHASE_LINE_COLORS[phase]),
                row=row, col=col
            )


def get_severity(d_total):
    if d_total < 0.3:   return 'NOMINAL',  '#2ecc71', 'nominal'
    elif d_total < 0.6: return 'MODERATE', '#f1c40f', 'moderate'
    elif d_total < 1.0: return 'HIGH',     '#e67e22', 'high'
    else:               return 'CRITICAL', '#e74c3c', 'critical'


def get_recommendation(d_total, top_features):
    n = len(top_features)
    if d_total < 0.3:
        return ("✅ No Action Required",
                "Your sepsis prediction model is performing within expected parameters. "
                "No feature distributions have shifted significantly from the training baseline.",
                "LOW", "Continue routine monitoring on the next data batch.")
    elif d_total < 0.6:
        feats = ", ".join(top_features[:3])
        return ("⚠️ Recalibration Recommended",
                f"Your sepsis model is showing moderate drift in {n} feature(s) ({feats}). "
                "Prediction confidence may be degrading. Recalibration with recent data is recommended.",
                "MEDIUM", "Recalibrate the model using the last 30 days of ICU records within 1-2 weeks.")
    elif d_total < 1.0:
        feats = ", ".join(top_features[:3])
        return ("🔴 Partial Retrain Required",
                f"Significant drift detected across {n} feature(s) ({feats}). "
                "Model reliability is compromised. Partial retraining on affected feature subsets is needed.",
                "HIGH", "Retrain the model on affected features using the last 60 days of data within 1 week.")
    else:
        return ("🚨 Full Retrain Urgent",
                f"Severe multi-system drift detected across {n} feature(s). "
                "The model is no longer reliable for clinical decision support. Full retraining is urgent.",
                "CRITICAL", "Immediately flag for clinical review. Full model retrain required before continued use.")


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING (for Review Mode)
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_results():
    p = PROJECT_ROOT / 'results' / 'drift2act_results.csv'
    return pd.read_csv(p) if p.exists() else None

@st.cache_data
def load_baseline_metrics():
    p = PROJECT_ROOT / 'results' / 'baseline_metrics.json'
    return json.load(open(p)) if p.exists() else {}

@st.cache_data
def load_alerts():
    p = PROJECT_ROOT / 'results' / 'alert_log.json'
    return json.load(open(p)) if p.exists() else []

@st.cache_data
def load_table(name):
    p = PROJECT_ROOT / 'results' / 'tables' / name
    return pd.read_csv(p) if p.exists() else None

@st.cache_data
def load_json(name):
    p = PROJECT_ROOT / 'results' / name
    return json.load(open(p)) if p.exists() else {}

@st.cache_data
def load_reference_data():
    p = PROJECT_ROOT / 'data' / 'processed' / 'patient_level.csv'
    return pd.read_csv(p) if p.exists() else None

@st.cache_data
def load_fingerprint():
    p = PROJECT_ROOT / 'results' / 'shap_fingerprint_phase1.pkl'
    return pickle.load(open(p, 'rb')) if p.exists() else None

@st.cache_data
def load_fingerprint_distributions():
    p = PROJECT_ROOT / 'results' / 'shap_fingerprint_phase1_distributions.pkl'
    return pickle.load(open(p, 'rb')) if p.exists() else None


# ═══════════════════════════════════════════════════════════════════════════
# UPLOAD & ANALYZE ENGINE
# ═══════════════════════════════════════════════════════════════════════════

def detect_format(df):
    """Detect if CSV is hourly (raw) or patient-level (aggregated)."""
    if 'HR' in df.columns and 'HR_mean' not in df.columns:
        return 'hourly'
    elif 'HR_mean' in df.columns:
        return 'patient_level'
    else:
        return 'unknown'


def aggregate_hourly_to_patient(df):
    """Aggregate hourly ICU data to patient-level features."""
    pid_col = None
    for c in ['Patient_ID', 'patient_id', 'PatientID', 'PATIENT_ID', 'pid']:
        if c in df.columns:
            pid_col = c
            break
    if pid_col is None:
        st.error("No Patient_ID column found.")
        return None

    clinical_features = [c for c in VITAL_FEATURES + LAB_FEATURES if c in df.columns]
    agg_dict = {}
    for feat in clinical_features:
        agg_dict[f'{feat}_mean'] = (feat, 'mean')
        agg_dict[f'{feat}_std']  = (feat, 'std')
        agg_dict[f'{feat}_min'] = (feat, 'min')
        agg_dict[f'{feat}_max'] = (feat, 'max')

    if 'SepsisLabel' in df.columns:
        agg_dict['SepsisLabel'] = ('SepsisLabel', 'max')
    if 'Age' in df.columns:
        agg_dict['Age'] = ('Age', 'first')
    if 'Gender' in df.columns:
        agg_dict['Gender'] = ('Gender', 'first')
    if 'ICULOS' in df.columns:
        agg_dict['ICULOS'] = ('ICULOS', 'max')

    result = df.groupby(pid_col).agg(**agg_dict).reset_index()
    result.rename(columns={pid_col: 'patient_id'}, inplace=True)
    return result


def run_drift_analysis(df_uploaded):
    """Run full drift analysis on uploaded data against saved baseline."""
    report = {'status': 'error', 'message': ''}

    # 1. Load baseline artifacts
    ref_data = load_reference_data()
    if ref_data is None:
        report['message'] = "No baseline reference data found. Run the pipeline first."
        return report

    fingerprint = load_fingerprint()
    baseline_metrics = load_baseline_metrics()

    # 2. Detect format and aggregate if needed
    fmt = detect_format(df_uploaded)
    if fmt == 'hourly':
        df_patient = aggregate_hourly_to_patient(df_uploaded)
        if df_patient is None:
            report['message'] = "Could not aggregate hourly data."
            return report
    elif fmt == 'patient_level':
        df_patient = df_uploaded.copy()
    else:
        report['message'] = "Unrecognized CSV format. Expected clinical columns like HR, O2Sat, etc."
        return report

    report['n_patients'] = len(df_patient)
    report['format'] = fmt

    # 3. Identify feature columns (intersection with reference)
    ref_feature_cols = [c for c in ref_data.columns if c.endswith(('_mean', '_std', '_min', '_max'))]
    upload_feature_cols = [c for c in df_patient.columns if c in ref_feature_cols]

    if len(upload_feature_cols) < 10:
        report['message'] = f"Only {len(upload_feature_cols)} matching features found. Need at least 10."
        return report

    report['n_features'] = len(upload_feature_cols)

    # 4. Prepare data
    ref_phase1 = ref_data[ref_data.get('phase', pd.Series(['phase1'] * len(ref_data))) == 'phase1']
    if len(ref_phase1) == 0:
        ref_phase1 = ref_data

    X_ref = ref_phase1[upload_feature_cols].copy()
    X_new = df_patient[upload_feature_cols].copy()

    # Fill NaN for analysis
    X_ref = X_ref.fillna(X_ref.median())
    X_new_filled = X_new.fillna(X_ref.median())  # Use ref medians for consistency

    # 5. KS Test per feature
    from scipy.stats import ks_2samp
    ks_results = {}
    ks_flagged = []
    for feat in upload_feature_cols:
        ref_vals = X_ref[feat].dropna().values
        new_vals = X_new_filled[feat].dropna().values
        if len(ref_vals) > 5 and len(new_vals) > 5:
            stat, pval = ks_2samp(ref_vals, new_vals)
            ks_results[feat] = {'statistic': float(stat), 'p_value': float(pval)}
            if pval < 0.05:
                ks_flagged.append((feat, stat, pval))

    ks_flagged.sort(key=lambda x: x[1], reverse=True)
    report['ks_results'] = ks_results
    report['ks_flagged'] = ks_flagged
    report['ks_n_flagged'] = len(ks_flagged)

    # 6. PSI per feature
    def compute_psi(ref_vals, new_vals, bins=10):
        eps = 1e-6
        combined = np.concatenate([ref_vals, new_vals])
        breakpoints = np.linspace(np.min(combined), np.max(combined), bins + 1)
        ref_pct = np.histogram(ref_vals, bins=breakpoints)[0] / len(ref_vals) + eps
        new_pct = np.histogram(new_vals, bins=breakpoints)[0] / len(new_vals) + eps
        return float(np.sum((new_pct - ref_pct) * np.log(new_pct / ref_pct)))

    psi_results = {}
    psi_flagged = []
    for feat in upload_feature_cols:
        ref_vals = X_ref[feat].dropna().values
        new_vals = X_new_filled[feat].dropna().values
        if len(ref_vals) > 5 and len(new_vals) > 5:
            psi = compute_psi(ref_vals, new_vals)
            psi_results[feat] = psi
            if psi > 0.2:
                psi_flagged.append((feat, psi))

    psi_flagged.sort(key=lambda x: x[1], reverse=True)
    report['psi_results'] = psi_results
    report['psi_flagged'] = psi_flagged

    # 7. SHAP + SADI analysis (if model & fingerprint available)
    model_path = PROJECT_ROOT / 'models' / 'xgb_calibrated.pkl'
    if not model_path.exists():
        model_path = PROJECT_ROOT / 'models' / 'xgb_baseline.pkl'

    sadi_scores = {}
    top_sadi_features = []

    if model_path.exists() and fingerprint is not None:
        try:
            model = joblib.load(model_path)
            # Get base estimator for SHAP
            base_model = model
            if hasattr(model, 'estimator'):
                base_model = model.estimator
            elif hasattr(model, 'estimators_'):
                base_model = model.estimators_[0]

            # Scale data using ref statistics
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            scaler.fit(X_ref[upload_feature_cols])
            X_new_scaled = pd.DataFrame(
                scaler.transform(X_new_filled[upload_feature_cols]),
                columns=upload_feature_cols
            )
            X_ref_scaled = pd.DataFrame(
                scaler.transform(X_ref[upload_feature_cols]),
                columns=upload_feature_cols
            )

            # Model predictions
            try:
                preds = model.predict_proba(X_new_scaled)[:, 1]
            except Exception:
                preds = base_model.predict_proba(X_new_scaled)[:, 1]
            report['predictions'] = preds

            ref_preds_path = PROJECT_ROOT / 'results' / 'ref_predictions.npy'
            if ref_preds_path.exists():
                ref_preds = np.load(ref_preds_path)
            else:
                try:
                    ref_preds = model.predict_proba(X_ref_scaled)[:, 1]
                except Exception:
                    ref_preds = base_model.predict_proba(X_ref_scaled)[:, 1]

            report['mean_pred'] = float(np.mean(preds))
            report['ref_mean_pred'] = float(np.mean(ref_preds))

            # SHAP (subsample for speed)
            import shap
            n_shap = min(100, len(X_new_scaled))
            X_shap = X_new_scaled.sample(n=n_shap, random_state=42) if len(X_new_scaled) > n_shap else X_new_scaled
            explainer = shap.TreeExplainer(base_model)
            shap_vals = explainer.shap_values(X_shap)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]

            # Build new fingerprint
            new_fp = {}
            for i, feat in enumerate(upload_feature_cols):
                vals = shap_vals[:, i]
                new_fp[feat] = {
                    'mean_abs': float(np.mean(np.abs(vals))),
                    'mean': float(np.mean(vals)),
                    'std': float(np.std(vals)),
                    'rank': 0
                }
            sorted_feats = sorted(new_fp.keys(), key=lambda f: new_fp[f]['mean_abs'], reverse=True)
            for rank, feat in enumerate(sorted_feats):
                new_fp[feat]['rank'] = rank

            # SADI computation
            for feat in upload_feature_cols:
                if feat in fingerprint and feat in new_fp:
                    ref_rank = fingerprint[feat].get('rank', 0)
                    new_rank = new_fp[feat]['rank']
                    ref_mean = fingerprint[feat].get('mean', 0)
                    new_mean = new_fp[feat].get('mean', 0)
                    ref_abs = fingerprint[feat].get('mean_abs', 0)
                    new_abs = new_fp[feat].get('mean_abs', 0)

                    # KL approximation using mean_abs difference
                    kl_approx = abs(new_abs - ref_abs) / max(ref_abs, 1e-6)
                    rank_shift = abs(new_rank - ref_rank) / max(len(upload_feature_cols), 1)
                    dir_flip = 1.0 if (ref_mean > 0) != (new_mean > 0) and abs(ref_mean) > 1e-4 else 0.0

                    sadi = 0.5 * kl_approx + 0.3 * rank_shift + 0.2 * dir_flip
                    sadi_scores[feat] = {
                        'sadi': sadi, 'kl': kl_approx,
                        'rank_shift': rank_shift, 'dir_flip': dir_flip,
                        'ref_importance': ref_abs, 'new_importance': new_abs
                    }

            top_sadi_features = sorted(sadi_scores.keys(), key=lambda f: sadi_scores[f]['sadi'], reverse=True)
            report['sadi_scores'] = sadi_scores
            report['top_sadi_features'] = top_sadi_features
            report['new_fingerprint'] = new_fp

            # Compute D_total
            top_n = min(10, len(top_sadi_features))
            d_shap = np.mean([sadi_scores[f]['sadi'] for f in top_sadi_features[:top_n]]) if top_n > 0 else 0
            d_feature = np.mean([psi_results.get(f, 0) for f in upload_feature_cols]) if psi_results else 0

            from scipy.stats import wasserstein_distance
            try:
                d_conf = wasserstein_distance(ref_preds, preds)
            except Exception:
                d_conf = abs(np.mean(preds) - np.mean(ref_preds))

            d_total = 0.5 * d_shap + 0.3 * d_feature + 0.2 * d_conf
            report['d_total'] = float(d_total)
            report['d_shap'] = float(d_shap)
            report['d_feature'] = float(d_feature)
            report['d_confidence'] = float(d_conf)
            report['has_shap'] = True

        except Exception as e:
            report['shap_error'] = str(e)
            report['has_shap'] = False
            # Fallback D_total from KS/PSI only
            mean_psi = np.mean(list(psi_results.values())) if psi_results else 0
            ks_frac = len(ks_flagged) / max(len(upload_feature_cols), 1)
            report['d_total'] = 0.6 * mean_psi + 0.4 * ks_frac
            report['d_shap'] = 0
            report['d_feature'] = mean_psi
            report['d_confidence'] = 0
    else:
        report['has_shap'] = False
        mean_psi = np.mean(list(psi_results.values())) if psi_results else 0
        ks_frac = len(ks_flagged) / max(len(upload_feature_cols), 1)
        report['d_total'] = 0.6 * mean_psi + 0.4 * ks_frac
        report['d_shap'] = 0
        report['d_feature'] = mean_psi
        report['d_confidence'] = 0

    # 8. Fairness (if Gender available)
    if 'Gender' in df_patient.columns and report.get('predictions') is not None:
        preds = report['predictions']
        threshold = 0.42
        binary_preds = (preds >= threshold).astype(int)
        g0 = df_patient['Gender'] == 0
        g1 = df_patient['Gender'] == 1
        if g0.sum() > 0 and g1.sum() > 0:
            rate0 = binary_preds[g0].mean()
            rate1 = binary_preds[g1].mean()
            report['dpd'] = float(abs(rate0 - rate1))
        else:
            report['dpd'] = 0.0
    else:
        report['dpd'] = None

    # Determine all top features (union of SADI + KS + PSI)
    all_flagged = set(top_sadi_features[:10])
    all_flagged.update([f[0] for f in ks_flagged[:10]])
    all_flagged.update([f[0] for f in psi_flagged[:10]])
    report['all_flagged_features'] = sorted(all_flagged)

    report['status'] = 'success'
    report['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report['baseline_auprc'] = baseline_metrics.get('xgboost', baseline_metrics.get('xgboost_calibrated', {})).get('auprc', 0)

    return report


# ═══════════════════════════════════════════════════════════════════════════
# UPLOAD MODE RENDERING
# ═══════════════════════════════════════════════════════════════════════════

def render_upload_mode():
    st.markdown("""
    <div class="main-header">
        <h1>🏥 Drift2Act</h1>
        <p>Upload patient records · Get a drift report · Know if your model is still reliable</p>
    </div>
    """, unsafe_allow_html=True)

    # Intro
    st.markdown("""
    <div class="upload-zone">
        <h3>📤 Upload Patient Records</h3>
        <p>Export your last month's ICU patient records as a CSV.<br>
        Drift2Act compares them against your model's training baseline and tells you what changed.</p>
    </div>
    """, unsafe_allow_html=True)

    # File format help
    with st.expander("📋 What format should my CSV be?", expanded=False):
        st.markdown("""
        **Option A — Hourly ICU records** (recommended)
        Each row = 1 hour for 1 patient. Must include:
        - `Patient_ID` — unique patient identifier
        - Clinical columns: `HR`, `O2Sat`, `Temp`, `SBP`, `MAP`, `DBP`, `Resp`, etc.
        - `SepsisLabel` (0/1) — optional but recommended
        - `Age`, `Gender` — for fairness analysis

        **Option B — Pre-aggregated patient records**
        Each row = 1 patient with summary statistics:
        - Columns like `HR_mean`, `HR_std`, `O2Sat_mean`, `O2Sat_max`, etc.
        - `SepsisLabel`, `Age`, `Gender`

        Drift2Act auto-detects the format.
        """)

    # Upload
    uploaded_file = st.file_uploader(
        "Drop your CSV here", type=['csv'],
        help="Standard clinical CSV with ICU patient records"
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            return

        # --- Data Preview ---
        fmt = detect_format(df)
        fmt_label = "Hourly ICU records" if fmt == 'hourly' else "Patient-level aggregated" if fmt == 'patient_level' else "Unknown format"

        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", f"{len(df):,}")
        col2.metric("Columns", len(df.columns))
        col3.metric("Format Detected", fmt_label)

        if fmt == 'hourly':
            pid_col = next((c for c in ['Patient_ID', 'patient_id', 'PatientID'] if c in df.columns), None)
            if pid_col:
                n_patients = df[pid_col].nunique()
                st.info(f"📊 Found **{n_patients}** unique patients across **{len(df):,}** hourly records. Will aggregate to patient-level before analysis.")

        with st.expander("👁️ Preview uploaded data", expanded=False):
            st.dataframe(df.head(20), use_container_width=True)

        if fmt == 'unknown':
            st.error("❌ Could not detect data format. Expected columns like `HR`, `O2Sat`, `Temp` (hourly) or `HR_mean`, `O2Sat_mean` (aggregated).")
            st.markdown(f"**Your columns:** `{', '.join(df.columns[:20])}`...")
            return

        # --- Analyze Button ---
        st.markdown("---")
        if st.button("🔍 Analyze for Drift", type="primary", use_container_width=True):
            with st.spinner("Analyzing data against baseline model... This may take 30-60 seconds."):
                report = run_drift_analysis(df)

            if report['status'] != 'success':
                st.error(f"❌ Analysis failed: {report.get('message', 'Unknown error')}")
                return

            # Save report to session state
            st.session_state['drift_report'] = report
            st.session_state['uploaded_filename'] = uploaded_file.name

        # --- Render Report ---
        if 'drift_report' in st.session_state:
            render_drift_report(st.session_state['drift_report'],
                              st.session_state.get('uploaded_filename', 'data.csv'))


def render_drift_report(report, filename):
    """Render the one-page drift analysis report."""
    d_total = report.get('d_total', 0)
    severity, sev_color, sev_class = get_severity(d_total)

    top_features = report.get('top_sadi_features', [f[0] for f in report.get('ks_flagged', [])])[:10]
    title, summary, priority, action = get_recommendation(d_total, top_features)

    # ── Status Banner ─────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="status-banner status-{sev_class}">
        <h2>{title}</h2>
        <p>{summary}</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Report Meta ───────────────────────────────────────────────────────
    st.caption(f"📄 Source: `{filename}` · {report.get('n_patients', '?')} patients · "
               f"{report.get('n_features', '?')} features · Analyzed: {report.get('timestamp', 'now')}")

    # ── KPI Cards ─────────────────────────────────────────────────────────
    d_shap = report.get('d_shap', 0)
    d_feat = report.get('d_feature', 0)
    d_conf = report.get('d_confidence', 0)

    st.markdown(f"""
    <div class="kpi-row">
        <div class="kpi-card">
            <div class="kpi-label">D_total</div>
            <div class="kpi-value" style="color:{sev_color}">{d_total:.3f}</div>
            <div class="kpi-delta" style="color:{sev_color}">● {severity}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">D_SHAP</div>
            <div class="kpi-value">{d_shap:.3f}</div>
            <div class="kpi-delta" style="color:#8888aa">Explanation drift</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">D_Feature</div>
            <div class="kpi-value">{d_feat:.3f}</div>
            <div class="kpi-delta" style="color:#8888aa">Distribution shift</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">D_Confidence</div>
            <div class="kpi-value">{d_conf:.3f}</div>
            <div class="kpi-delta" style="color:#8888aa">Prediction shift</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Features Flagged</div>
            <div class="kpi-value">{report.get('ks_n_flagged', 0)}</div>
            <div class="kpi-delta" style="color:#8888aa">of {report.get('n_features', '?')}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Top Drifted Features ──────────────────────────────────────────────
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### 🔥 Top Drifted Features")

        sadi_scores = report.get('sadi_scores', {})
        ks_flagged = report.get('ks_flagged', [])

        if sadi_scores:
            top = sorted(sadi_scores.items(), key=lambda x: x[1]['sadi'], reverse=True)[:12]
            feat_names = [f[0].replace('_mean', '').replace('_std', ' σ').replace('_min', ' ↓').replace('_max', ' ↑') for f in top]
            feat_sadi = [f[1]['sadi'] for f in top]

            fig = go.Figure(go.Bar(
                y=feat_names, x=feat_sadi, orientation='h',
                marker=dict(color=feat_sadi, colorscale='YlOrRd', showscale=True,
                           colorbar=dict(title="SADI", len=0.6)),
                text=[f"{v:.3f}" for v in feat_sadi], textposition='outside'
            ))
            fig.update_layout(
                title="SADI Score per Feature",
                height=420, template="plotly_dark",
                yaxis=dict(autorange="reversed"),
                margin=dict(l=140, r=60, t=50, b=30),
                xaxis_title="SADI Score"
            )
            st.plotly_chart(fig, use_container_width=True)
        elif ks_flagged:
            feat_names = [f[0] for f in ks_flagged[:12]]
            feat_stats = [f[1] for f in ks_flagged[:12]]
            fig = go.Figure(go.Bar(
                y=feat_names, x=feat_stats, orientation='h',
                marker=dict(color=feat_stats, colorscale='YlOrRd'),
                text=[f"{v:.3f}" for v in feat_stats], textposition='outside'
            ))
            fig.update_layout(
                title="KS Statistic per Feature (higher = more drift)",
                height=420, template="plotly_dark",
                yaxis=dict(autorange="reversed"),
                margin=dict(l=140, r=60, t=50, b=30)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No significant feature-level drift detected.")

    with col2:
        st.markdown("#### 📊 Distribution Shift (KS Test)")

        if ks_flagged:
            ks_df = pd.DataFrame(ks_flagged, columns=['Feature', 'KS Statistic', 'p-value'])
            ks_df['Significant'] = ks_df['p-value'].apply(lambda p: '✅ Yes' if p < 0.05 else '❌ No')
            ks_df['KS Statistic'] = ks_df['KS Statistic'].round(4)
            ks_df['p-value'] = ks_df['p-value'].apply(lambda p: f"{p:.2e}")
            st.dataframe(ks_df.head(15), use_container_width=True, height=380)
        else:
            st.success("✅ No features show statistically significant distribution shift (KS test, p < 0.05).")

    # ── SHAP Importance Comparison ─────────────────────────────────────────
    if report.get('has_shap') and report.get('new_fingerprint'):
        st.markdown("---")
        st.markdown("#### 🧬 SHAP Feature Importance: Baseline vs Uploaded Data")

        fp_ref = load_fingerprint()
        fp_new = report['new_fingerprint']

        if fp_ref:
            common = sorted(set(fp_ref.keys()) & set(fp_new.keys()),
                          key=lambda f: fp_ref[f].get('mean_abs', 0), reverse=True)[:15]

            ref_vals = [fp_ref[f].get('mean_abs', 0) for f in common]
            new_vals = [fp_new[f].get('mean_abs', 0) for f in common]
            labels = [f.replace('_mean', '').replace('_std', ' σ') for f in common]

            fig = go.Figure()
            fig.add_trace(go.Bar(name='Baseline (Training)', y=labels, x=ref_vals,
                                orientation='h', marker_color='#3498db'))
            fig.add_trace(go.Bar(name='Uploaded Batch', y=labels, x=new_vals,
                                orientation='h', marker_color='#e74c3c'))
            fig.update_layout(
                barmode='group', height=450, template="plotly_dark",
                title="Mean |SHAP| — Top 15 Features",
                yaxis=dict(autorange="reversed"),
                margin=dict(l=140, r=30, t=50, b=30),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"),
                xaxis_title="Mean |SHAP Value|"
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── PSI Analysis ──────────────────────────────────────────────────────
    psi_results = report.get('psi_results', {})
    if psi_results:
        with st.expander("📈 Population Stability Index (PSI) Details", expanded=False):
            psi_df = pd.DataFrame([
                {'Feature': k, 'PSI': round(v, 4),
                 'Status': '🔴 Major shift' if v > 0.25 else '🟡 Moderate shift' if v > 0.1 else '🟢 Stable'}
                for k, v in sorted(psi_results.items(), key=lambda x: x[1], reverse=True)
            ])
            st.dataframe(psi_df.head(20), use_container_width=True)

    # ── Fairness ──────────────────────────────────────────────────────────
    if report.get('dpd') is not None:
        st.markdown("---")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("#### ⚖️ Fairness Check")
            dpd = report['dpd']
            fair_status = "✅ Fair" if dpd < 0.1 else "⚠️ Potential bias"
            fair_color = "#2ecc71" if dpd < 0.1 else "#e74c3c"
            st.metric("Demographic Parity Diff", f"{dpd:.4f}", delta=fair_status)
        with col2:
            st.markdown("#### 📊 Prediction Distribution")
            preds = report.get('predictions')
            if preds is not None:
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=preds, nbinsx=30, name="Predicted P(sepsis)",
                                          marker_color='#3498db', opacity=0.8))
                fig.add_vline(x=0.42, line_dash="dash", line_color="#e74c3c",
                            annotation_text="Decision threshold (0.42)")
                fig.update_layout(height=250, template="plotly_dark",
                                margin=dict(l=30, r=30, t=30, b=30),
                                xaxis_title="P(sepsis)", yaxis_title="Count")
                st.plotly_chart(fig, use_container_width=True)

    # ── Recommendation Box ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"""
    <div class="rec-box" style="border-left-color: {sev_color};">
        <h4>📋 Recommendation</h4>
        <p><strong>Priority:</strong> {priority} &nbsp;|&nbsp; <strong>Action:</strong> {action}</p>
        <p style="margin-top:0.8rem; color:#aaa; font-size:0.85rem;">
            This report was generated by Drift2Act v1.0 using {report.get('n_patients', '?')} patient records
            compared against the training baseline
            {'(SHAP + SADI + KS + PSI analysis)' if report.get('has_shap') else '(KS + PSI analysis only — model not available for SHAP)'}.
            {'Baseline model AUPRC: ' + f"{report.get('baseline_auprc', 0):.4f}" if report.get('baseline_auprc') else ''}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # SHAP error note
    if report.get('shap_error'):
        with st.expander("⚠️ SHAP analysis encountered an error"):
            st.code(report['shap_error'])
            st.info("The report used KS + PSI statistical tests instead. Results are still valid but less granular.")


# ═══════════════════════════════════════════════════════════════════════════
# REVIEW MODE RENDERING
# ═══════════════════════════════════════════════════════════════════════════

def render_review_mode():
    results_df = load_results()
    baseline_metrics = load_baseline_metrics()
    alerts = load_alerts()
    comparison_table = load_table('table1_detector_comparison.csv')
    ablation_table = load_table('table3_ablation.csv')
    fairness_table = load_table('table4_fairness.csv')
    attribution_table = load_table('table2_attribution_accuracy.csv')
    drift_gt = load_json('drift_ground_truth.json')
    stat_sig = load_json('statistical_significance.json')

    st.markdown("""
    <div class="main-header">
        <h1>🏥 Drift2Act — Pipeline Review</h1>
        <p>Detailed analysis from the last full pipeline run</p>
    </div>
    """, unsafe_allow_html=True)

    if results_df is None:
        st.error("⚠️ No pipeline results found. Run `python scripts/run_pipeline.py` first.")
        st.info("Or switch to **Upload & Analyze** mode to test with a CSV file.")
        return

    # ── Sidebar controls ─────────────────────────────────────────────────
    sadi_threshold = st.sidebar.slider("SADI Threshold", 0.1, 2.0, 0.30, 0.05)
    window_range = st.sidebar.slider("Window Range", 0, int(results_df['window_num'].max()),
                                     (0, int(results_df['window_num'].max())))
    phase_filter = st.sidebar.multiselect("Phase Filter", ['phase1', 'phase2', 'phase3'],
                                          default=['phase1', 'phase2', 'phase3'])

    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Run Summary")
    for phase, count in sorted(results_df['true_phase'].value_counts().items()):
        emoji = {'phase1': '🟢', 'phase2': '🟡', 'phase3': '🔴'}.get(phase, '⚪')
        st.sidebar.markdown(f"{emoji} **{phase}**: {count} windows")
    if baseline_metrics:
        xgb = baseline_metrics.get('xgboost', baseline_metrics.get('xgboost_calibrated', {}))
        st.sidebar.markdown(f"**Baseline AUPRC:** {xgb.get('auprc', 0):.4f}")

    # ── Filter ───────────────────────────────────────────────────────────
    mask = ((results_df['window_num'] >= window_range[0]) &
            (results_df['window_num'] <= window_range[1]) &
            (results_df['true_phase'].isin(phase_filter)))
    filtered_df = results_df[mask].copy()

    # ── KPIs ─────────────────────────────────────────────────────────────
    if len(filtered_df) > 0:
        latest = filtered_df.iloc[-1]
        d_total = latest['D_total']
        severity, sev_color, _ = get_severity(d_total)
        level = int(latest['intervention_level'])
        level_names = {0: 'No Action', 1: 'Alert', 2: 'Recalibrate', 3: 'Partial Retrain', 4: 'Full Retrain'}
        level_colors = {0: '#2ecc71', 1: '#3498db', 2: '#f1c40f', 3: '#e67e22', 4: '#e74c3c'}
        est_auprc = latest.get('est_auprc', 0)
        baseline = latest.get('baseline_auprc', 0)
        perf_drop = baseline - est_auprc if baseline > 0 else 0

        st.markdown(f"""
        <div class="kpi-row">
            <div class="kpi-card">
                <div class="kpi-label">D_total (Latest)</div>
                <div class="kpi-value" style="color:{sev_color}">{d_total:.3f}</div>
                <div class="kpi-delta" style="color:{sev_color}">● {severity}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Intervention</div>
                <div class="kpi-value" style="color:{level_colors[level]}">L{level}</div>
                <div class="kpi-delta" style="color:{level_colors[level]}">{level_names[level]}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Est. AUPRC</div>
                <div class="kpi-value">{est_auprc:.3f}</div>
                <div class="kpi-delta" style="color:{'#e74c3c' if perf_drop>0.02 else '#2ecc71'}">{'▼' if perf_drop>0 else '▲'} {perf_drop:.3f}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Risk Score</div>
                <div class="kpi-value">{latest['risk_score']:.3f}</div>
                <div class="kpi-delta" style="color:#8888aa">Perf. est.</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Drift Belief</div>
                <div class="kpi-value">{latest.get('drift_belief',0):.3f}</div>
                <div class="kpi-delta" style="color:#8888aa">Composite</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Tabs ─────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📈 SADI Timeline", "🔥 Feature Drift", "🧬 SHAP",
        "🎯 Interventions", "⚖️ Fairness", "📊 Detectors", "🚨 Alerts"
    ])

    # ── Tab 1: SADI Timeline ─────────────────────────────────────────────
    with tab1:
        st.subheader("SADI Drift Score Over Time")
        fig = make_subplots(rows=3, cols=1, row_heights=[0.5, 0.25, 0.25],
                           shared_xaxes=True, vertical_spacing=0.06,
                           subplot_titles=("D_total", "D_SHAP Component", "Adversarial Score"))
        for r in [1, 2, 3]:
            add_phase_shading(fig, filtered_df, row=r, col=1)

        fig.add_trace(go.Scatter(x=filtered_df['window_num'], y=filtered_df['D_total'],
                                mode='lines', name='D_total',
                                line=dict(color='#3498db', width=3),
                                fill='tozeroy', fillcolor='rgba(52,152,219,0.08)'), row=1, col=1)
        fig.add_hline(y=sadi_threshold, line_dash="dash", line_color="#e74c3c",
                      annotation_text=f"Threshold ({sadi_threshold})", row=1, col=1)
        fig.add_trace(go.Scatter(x=filtered_df['window_num'], y=filtered_df['D_shap'],
                                mode='lines', name='D_SHAP',
                                line=dict(color='#9b59b6', width=1.5)), row=2, col=1)
        fig.add_trace(go.Scatter(x=filtered_df['window_num'], y=filtered_df['adv_score'],
                                mode='lines+markers', name='Adv. AUROC',
                                line=dict(color='#e67e22', width=2),
                                marker=dict(size=4)), row=3, col=1)
        fig.add_hline(y=0.5, line_dash="dash", line_color="#555", row=3, col=1)
        fig.update_layout(height=650, template="plotly_dark",
                         legend=dict(orientation="h", yanchor="bottom", y=1.03, x=0.5, xanchor="center"),
                         margin=dict(l=50, r=30, t=70, b=30))
        st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mean D_total", f"{filtered_df['D_total'].mean():.3f}")
        c2.metric("Max D_total", f"{filtered_df['D_total'].max():.3f}")
        c3.metric("Windows > Thresh", f"{(filtered_df['D_total'] > sadi_threshold).sum()}/{len(filtered_df)}")
        c4.metric("Mean Adv.", f"{filtered_df['adv_score'].mean():.3f}")

    # ── Tab 2: Feature Drift ─────────────────────────────────────────────
    with tab2:
        st.subheader("Feature-Level Drift Analysis")
        try:
            feature_data = []
            for _, row in filtered_df.iterrows():
                feats = row.get('top_sadi_features', '[]')
                if isinstance(feats, str):
                    try: feats = json.loads(feats)
                    except: feats = []
                for rank, feat in enumerate(feats):
                    feature_data.append({'window': row['window_num'], 'feature': feat, 'rank': rank+1, 'phase': row['true_phase']})

            if feature_data:
                feat_df = pd.DataFrame(feature_data)
                c1, c2 = st.columns(2)
                with c1:
                    freq = feat_df['feature'].value_counts().head(15)
                    fig = go.Figure(go.Bar(x=freq.values, y=freq.index, orientation='h',
                                         marker=dict(color=freq.values, colorscale='YlOrRd', showscale=True),
                                         text=freq.values, textposition='outside'))
                    fig.update_layout(title="Most Flagged Features", height=450, template="plotly_dark",
                                    yaxis=dict(autorange="reversed"), margin=dict(l=160, r=50))
                    st.plotly_chart(fig, use_container_width=True)
                with c2:
                    top_feats = freq.head(10).index.tolist()
                    hm = feat_df[feat_df['feature'].isin(top_feats)].pivot_table(
                        index='feature', columns='window', values='rank', aggfunc='min', fill_value=0).reindex(top_feats)
                    fig = go.Figure(go.Heatmap(z=hm.values, x=[f"W{c}" for c in hm.columns], y=hm.index,
                                             colorscale=[[0,'#1a1a2e'],[0.2,'#2ecc71'],[0.5,'#f1c40f'],[1,'#e74c3c']],
                                             text=hm.values, texttemplate="%{text}"))
                    fig.update_layout(title="SADI Rank Heatmap", height=450, template="plotly_dark", margin=dict(l=160))
                    st.plotly_chart(fig, use_container_width=True)

                if drift_gt:
                    st.markdown("#### 🎯 Ground Truth vs Detection")
                    gt_f = set(drift_gt.keys()); det_f = set(freq.index)
                    tp = gt_f & det_f; fp = det_f - gt_f; fn = gt_f - det_f
                    c1, c2, c3 = st.columns(3)
                    c1.metric("True Positives", len(tp)); c2.metric("False Positives", len(fp)); c3.metric("Missed", len(fn))
                    if tp: st.success(f"Detected: {', '.join(sorted(tp))}")
                    if fn: st.warning(f"Missed: {', '.join(sorted(fn))}")
        except Exception as e:
            st.error(f"Error: {e}")

    # ── Tab 3: SHAP ──────────────────────────────────────────────────────
    with tab3:
        st.subheader("SHAP Feature Importance")
        c1, c2 = st.columns(2)
        with c1:
            p = PROJECT_ROOT / 'paper' / 'figures' / 'fig1_shap_beeswarm.png'
            if p.exists(): st.image(str(p), caption="SHAP Beeswarm — Phase 1 Baseline")
            else: st.info("SHAP beeswarm not yet generated.")
        with c2:
            p = PROJECT_ROOT / 'paper' / 'figures' / 'fig1_feature_distributions.png'
            if p.exists(): st.image(str(p), caption="Feature Distributions Across Phases")
            else: st.info("Feature distribution plot not yet generated.")

        if attribution_table is not None:
            st.markdown("---")
            st.markdown("#### 📐 Attribution Accuracy")
            det_col = 'Detector' if 'Detector' in attribution_table.columns else 'Method'
            fig = go.Figure()
            for i, m in enumerate(['Precision', 'Recall']):
                if m in attribution_table.columns:
                    fig.add_trace(go.Bar(name=m, x=attribution_table[det_col], y=attribution_table[m],
                                       marker_color=['#3498db','#2ecc71'][i],
                                       text=attribution_table[m].round(3), textposition='outside'))
            fig.update_layout(barmode='group', height=350, template="plotly_dark", yaxis=dict(range=[0,1.1]))
            st.plotly_chart(fig, use_container_width=True)

    # ── Tab 4: Interventions ─────────────────────────────────────────────
    with tab4:
        st.subheader("Intervention Log")
        lc = ['#2ecc71', '#3498db', '#f1c40f', '#e67e22', '#e74c3c']
        ln = {0:'No Action', 1:'Alert', 2:'Recalibrate', 3:'Partial Retrain', 4:'Full Retrain'}
        c1, c2 = st.columns([1, 2])
        with c1:
            lv = filtered_df['intervention_level'].value_counts().sort_index()
            fig = go.Figure(go.Pie(labels=[ln.get(int(i),str(i)) for i in lv.index], values=lv.values,
                                  marker=dict(colors=[lc[int(i)] for i in lv.index]), hole=0.55,
                                  textinfo='label+value'))
            fig.update_layout(title="Distribution", height=350, template="plotly_dark", showlegend=False,
                            annotations=[dict(text=f"{len(filtered_df)}<br>win", x=0.5, y=0.5, font_size=14, showarrow=False, font_color="white")])
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = go.Figure(go.Scatter(x=filtered_df['window_num'], y=filtered_df['intervention_level'],
                                      mode='markers+lines',
                                      marker=dict(color=[lc[int(v)] for v in filtered_df['intervention_level']], size=8),
                                      line=dict(color='#555', width=1)))
            fig.update_layout(title="Level Over Time", xaxis_title="Window", yaxis_title="Level",
                            yaxis=dict(tickvals=[0,1,2,3,4], ticktext=['None','Alert','Recal.','Partial','Full']),
                            height=350, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

        fig = px.scatter(filtered_df, x='D_total', y='est_auprc', color='true_phase',
                        color_discrete_map={'phase1':'#2ecc71','phase2':'#f1c40f','phase3':'#e74c3c'},
                        size='risk_score', size_max=12, hover_data=['window_num','intervention_action'],
                        template='plotly_dark', height=350)
        fig.update_layout(xaxis_title="D_total", yaxis_title="Est. AUPRC", margin=dict(t=30))
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab 5: Fairness ──────────────────────────────────────────────────
    with tab5:
        st.subheader("Fairness Monitoring")
        fig = make_subplots(rows=1, cols=2, subplot_titles=("DPD", "EOD"))
        add_phase_shading(fig, filtered_df, 1, 1); add_phase_shading(fig, filtered_df, 1, 2)
        fig.add_trace(go.Scatter(x=filtered_df['window_num'], y=filtered_df['dpd'], mode='lines+markers',
                                name='DPD', line=dict(color='#e74c3c', width=2), marker=dict(size=5)), row=1, col=1)
        fig.add_hline(y=0.10, line_dash="dash", line_color="#f39c12", row=1, col=1)
        fig.add_trace(go.Scatter(x=filtered_df['window_num'], y=filtered_df['eod'], mode='lines+markers',
                                name='EOD', line=dict(color='#3498db', width=2), marker=dict(size=5)), row=1, col=2)
        fig.add_hline(y=0.10, line_dash="dash", line_color="#f39c12", row=1, col=2)
        fig.update_layout(height=400, template="plotly_dark", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Mean DPD", f"{filtered_df['dpd'].mean():.4f}")
        c2.metric("Max DPD", f"{filtered_df['dpd'].max():.4f}")
        c3.metric("Mean EOD", f"{filtered_df['eod'].mean():.4f}")
        c4.metric("Max EOD", f"{filtered_df['eod'].max():.4f}")
        if fairness_table is not None:
            with st.expander("Fairness by Phase"):
                st.dataframe(fairness_table, use_container_width=True)

    # ── Tab 6: Detectors ─────────────────────────────────────────────────
    with tab6:
        st.subheader("Drift Detector Comparison")
        if comparison_table is not None:
            fig = go.Figure()
            for m, c in [('Precision','#3498db'),('Recall','#2ecc71'),('F1','#e74c3c')]:
                if m in comparison_table.columns:
                    fig.add_trace(go.Bar(name=m, x=comparison_table['Detector'], y=comparison_table[m],
                                       marker_color=c, text=comparison_table[m].round(3), textposition='outside',
                                       textfont=dict(size=13, color='white')))
            fig.update_layout(barmode='group', height=420, template="plotly_dark", yaxis=dict(range=[0,1.15]),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"))
            st.plotly_chart(fig, use_container_width=True)
            c1, c2 = st.columns(2)
            with c1: st.dataframe(comparison_table, use_container_width=True)
            with c2:
                if attribution_table is not None: st.dataframe(attribution_table, use_container_width=True)
        else:
            st.info("Run pipeline first.")

        if ablation_table is not None:
            st.markdown("---")
            st.markdown("#### 🔬 SADI Ablation")
            cfg_col = next((c for c in ablation_table.columns if c.lower().startswith('config')), ablation_table.columns[0])
            f1_col = 'F1' if 'F1' in ablation_table.columns else ablation_table.columns[-1]
            n_bars = len(ablation_table)
            colors = ['#95a5a6'] * (n_bars - 1) + ['#3498db']
            fig = go.Figure(go.Bar(x=ablation_table[cfg_col], y=ablation_table[f1_col],
                                  marker=dict(color=colors, line=dict(width=1.5, color='white')),
                                  text=ablation_table[f1_col].round(3), textposition='outside'))
            fig.update_layout(yaxis=dict(range=[0,1.1]), height=380, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

        if stat_sig:
            st.markdown("---")
            c1,c2,c3 = st.columns(3)
            c1.metric("SADI F1", f"{stat_sig.get('sadi_f1_mean',0):.3f} ± {stat_sig.get('sadi_f1_std',0):.3f}")
            c2.metric("KS F1", f"{stat_sig.get('ks_f1_mean',0):.3f} ± {stat_sig.get('ks_f1_std',0):.3f}")
            c3.metric("Wilcoxon p", f"{stat_sig.get('wilcoxon_p_value',1):.4f}",
                      delta="✓ Sig." if stat_sig.get('significant') else "Not sig.")

    # ── Tab 7: Alerts ────────────────────────────────────────────────────
    with tab7:
        st.subheader("Clinical Drift Alerts")
        if alerts:
            sev_f = st.selectbox("Min Level", [1,2,3,4], format_func=lambda x: {1:'🔵 L1',2:'🟡 L2',3:'🟠 L3',4:'🔴 L4'}[x])
            fa = [a for a in alerts if a.get('level',0) >= sev_f]
            st.markdown(f"**{len(fa)}/{len(alerts)} alerts**")
            for a in fa[-15:]:
                lv = a.get('level',0)
                ic = {1:'🔵',2:'🟡',3:'🟠',4:'🔴'}.get(lv,'⚪')
                with st.expander(f"{ic} W{a.get('window_idx',a.get('window_num','?'))} — {a.get('action','N/A')}"):
                    if 'alert_text' in a: st.markdown(f'<div class="alert-box">{a["alert_text"]}</div>', unsafe_allow_html=True)
                    else: st.json(a)
        else:
            st.markdown('<div style="text-align:center;padding:3rem;color:#888"><div style="font-size:3rem">✅</div>'
                       '<div style="font-size:1.2rem;font-weight:600">No Alerts</div></div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# PIPELINE UPLOAD MODE (TECHNICAL)
# ═══════════════════════════════════════════════════════════════════════════

def parse_pipeline_files(uploaded_files):
    """Parse uploaded pipeline result files into a data dict."""
    data = {}
    for f in uploaded_files:
        name = f.name.lower()
        content = f.read()
        f.seek(0)
        try:
            if name == 'drift2act_results.csv':
                data['results'] = pd.read_csv(f)
            elif name == 'baseline_metrics.json':
                data['baseline'] = json.loads(content)
            elif name == 'alert_log.json':
                data['alerts'] = json.loads(content)
            elif name == 'drift_ground_truth.json':
                data['drift_gt'] = json.loads(content)
            elif name == 'statistical_significance.json':
                data['stat_sig'] = json.loads(content)
            elif name == 'table1_detector_comparison.csv':
                data['table1'] = pd.read_csv(f)
            elif name == 'table2_attribution_accuracy.csv':
                data['table2'] = pd.read_csv(f)
            elif name == 'table3_ablation.csv':
                data['table3'] = pd.read_csv(f)
            elif name == 'table4_fairness.csv':
                data['table4'] = pd.read_csv(f)
        except Exception as e:
            st.warning(f"Could not parse {f.name}: {e}")
    return data


def parse_zip_pipeline(uploaded_zip):
    """Parse a ZIP archive containing pipeline results."""
    import zipfile, io
    data = {}
    with zipfile.ZipFile(io.BytesIO(uploaded_zip.read()), 'r') as zf:
        for info in zf.infolist():
            basename = info.filename.split('/')[-1].lower()
            if info.is_dir() or not basename:
                continue
            raw = zf.read(info.filename)
            try:
                if basename == 'drift2act_results.csv':
                    data['results'] = pd.read_csv(io.BytesIO(raw))
                elif basename == 'baseline_metrics.json':
                    data['baseline'] = json.loads(raw)
                elif basename == 'alert_log.json':
                    data['alerts'] = json.loads(raw)
                elif basename == 'drift_ground_truth.json':
                    data['drift_gt'] = json.loads(raw)
                elif basename == 'statistical_significance.json':
                    data['stat_sig'] = json.loads(raw)
                elif basename == 'table1_detector_comparison.csv':
                    data['table1'] = pd.read_csv(io.BytesIO(raw))
                elif basename == 'table2_attribution_accuracy.csv':
                    data['table2'] = pd.read_csv(io.BytesIO(raw))
                elif basename == 'table3_ablation.csv':
                    data['table3'] = pd.read_csv(io.BytesIO(raw))
                elif basename == 'table4_fairness.csv':
                    data['table4'] = pd.read_csv(io.BytesIO(raw))
            except Exception as e:
                st.warning(f"Could not parse {info.filename}: {e}")
    return data


def render_pipeline_upload_mode():
    st.markdown("""
    <div class="main-header">
        <h1>🔧 Pipeline Analysis — Technical View</h1>
        <p>Upload pipeline output files for full diagnostic analysis · For MLOps engineers and clinical data scientists</p>
    </div>
    """, unsafe_allow_html=True)

    # Upload section
    upload_method = st.radio("Upload method", ["📦 ZIP archive", "📄 Individual files"], horizontal=True)

    pdata = None
    if upload_method == "📦 ZIP archive":
        st.markdown("Upload a ZIP containing pipeline outputs (`drift2act_results.csv`, `baseline_metrics.json`, tables, etc.)")
        zf = st.file_uploader("Upload pipeline ZIP", type=['zip'], key='pipeline_zip')
        if zf:
            pdata = parse_zip_pipeline(zf)
    else:
        st.markdown("Upload individual pipeline output files. At minimum, `drift2act_results.csv` is required.")
        files = st.file_uploader("Upload pipeline files", type=['csv', 'json'],
                                accept_multiple_files=True, key='pipeline_files')
        if files:
            pdata = parse_pipeline_files(files)

    if pdata is None or 'results' not in pdata:
        with st.expander("📋 Expected files", expanded=False):
            st.markdown("""
            | File | Required | Description |
            |------|----------|-------------|
            | `drift2act_results.csv` | ✅ Yes | Per-window drift scores, interventions, fairness |
            | `baseline_metrics.json` | Optional | Model AUPRC, AUROC, Brier score |
            | `alert_log.json` | Optional | Clinical alert history |
            | `drift_ground_truth.json` | Optional | Known drifted features |
            | `statistical_significance.json` | Optional | Bootstrap test results |
            | `table1_detector_comparison.csv` | Optional | SADI vs KS vs PSI |
            | `table2_attribution_accuracy.csv` | Optional | Feature attribution |
            | `table3_ablation.csv` | Optional | SADI component ablation |
            | `table4_fairness.csv` | Optional | Phase-level fairness |
            """)
        return

    # ── Parse data ────────────────────────────────────────────────────────
    df = pdata['results']
    baseline = pdata.get('baseline', {})
    alerts = pdata.get('alerts', [])
    drift_gt = pdata.get('drift_gt', {})
    stat_sig = pdata.get('stat_sig', {})
    t1 = pdata.get('table1')
    t2 = pdata.get('table2')
    t3 = pdata.get('table3')
    t4 = pdata.get('table4')

    st.success(f"✅ Loaded pipeline: **{len(df)} windows** · "
               f"{df['true_phase'].nunique()} phases · "
               f"{len(pdata)} file(s) parsed")

    # ── Sidebar controls ──────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🔧 Pipeline Controls")
        thresh = st.slider("Drift Threshold", 0.1, 2.0, 0.30, 0.05, key='p_thresh')
        wrange = st.slider("Window Range", 0, int(df['window_num'].max()),
                          (0, int(df['window_num'].max())), key='p_wrange')
        phases = st.multiselect("Phases", df['true_phase'].unique().tolist(),
                               default=df['true_phase'].unique().tolist(), key='p_phases')
        st.markdown("---")
        st.markdown("### 📋 Loaded Files")
        file_icons = {'results': '📊', 'baseline': '🧠', 'alerts': '🚨',
                     'drift_gt': '🎯', 'stat_sig': '📈', 'table1': '📋',
                     'table2': '📋', 'table3': '📋', 'table4': '📋'}
        for k in pdata:
            st.markdown(f"{file_icons.get(k, '📄')} `{k}`")

    fdf = df[(df['window_num'] >= wrange[0]) & (df['window_num'] <= wrange[1]) &
             (df['true_phase'].isin(phases))].copy()

    if len(fdf) == 0:
        st.warning("No data in selected range.")
        return

    # ═════════════════════════════════════════════════════════════════════
    # KPI ROW
    # ═════════════════════════════════════════════════════════════════════
    latest = fdf.iloc[-1]
    d_total = latest['D_total']
    sev, sev_color, sev_class = get_severity(d_total)
    lv = int(latest['intervention_level'])
    ln = {0: 'NO_ACTION', 1: 'ALERT', 2: 'RECALIBRATE', 3: 'PARTIAL_RETRAIN', 4: 'FULL_RETRAIN'}
    lc = {0: '#2ecc71', 1: '#3498db', 2: '#f1c40f', 3: '#e67e22', 4: '#e74c3c'}
    xgb = baseline.get('xgboost', baseline.get('xgboost_calibrated', {}))

    st.markdown(f"""
    <div class="kpi-row">
        <div class="kpi-card">
            <div class="kpi-label">D_total (Latest)</div>
            <div class="kpi-value" style="color:{sev_color}">{d_total:.4f}</div>
            <div class="kpi-delta" style="color:{sev_color}">● {sev}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">D_SHAP</div>
            <div class="kpi-value">{latest['D_shap']:.4f}</div>
            <div class="kpi-delta" style="color:#8888aa">Explanation component</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Adversarial AUROC</div>
            <div class="kpi-value">{latest['adv_score']:.4f}</div>
            <div class="kpi-delta" style="color:#8888aa">Domain classifier</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Intervention</div>
            <div class="kpi-value" style="color:{lc[lv]}">L{lv}</div>
            <div class="kpi-delta" style="color:{lc[lv]}">{ln[lv]}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Baseline AUPRC</div>
            <div class="kpi-value">{xgb.get('auprc', latest.get('baseline_auprc', 0)):.4f}</div>
            <div class="kpi-delta" style="color:#8888aa">Reference</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ═════════════════════════════════════════════════════════════════════
    # TABS
    # ═════════════════════════════════════════════════════════════════════
    tabs = st.tabs([
        "📈 Overview", "🔬 Per-Window Data", "📐 SADI Components",
        "🔥 Feature Attribution", "🧠 Model Diagnostics",
        "📊 Detector Comparison", "⚖️ Fairness Audit", "🚨 Alert Timeline"
    ])

    # ── Tab: Overview ─────────────────────────────────────────────────────
    with tabs[0]:
        st.subheader("Pipeline Overview")

        fig = make_subplots(rows=3, cols=1, row_heights=[0.45, 0.30, 0.25],
                           shared_xaxes=True, vertical_spacing=0.06,
                           subplot_titles=("D_total Composite", "D_SHAP + Adversarial", "Intervention Level"))
        for r in [1, 2, 3]:
            add_phase_shading(fig, fdf, row=r, col=1)

        fig.add_trace(go.Scatter(x=fdf['window_num'], y=fdf['D_total'], mode='lines',
                                name='D_total', line=dict(color='#3498db', width=3),
                                fill='tozeroy', fillcolor='rgba(52,152,219,0.08)'), row=1, col=1)
        fig.add_hline(y=thresh, line_dash="dash", line_color="#e74c3c",
                      annotation_text=f"Threshold ({thresh})", row=1, col=1)

        fig.add_trace(go.Scatter(x=fdf['window_num'], y=fdf['D_shap'], mode='lines',
                                name='D_SHAP', line=dict(color='#9b59b6', width=2)), row=2, col=1)
        fig.add_trace(go.Scatter(x=fdf['window_num'], y=fdf['adv_score'], mode='lines',
                                name='Adv. AUROC', line=dict(color='#e67e22', width=2, dash='dot')), row=2, col=1)
        fig.add_hline(y=0.5, line_dash="dash", line_color="#555", row=2, col=1)

        int_colors = [lc.get(int(v), '#888') for v in fdf['intervention_level']]
        fig.add_trace(go.Bar(x=fdf['window_num'], y=fdf['intervention_level'],
                            name='Intervention', marker_color=int_colors), row=3, col=1)
        fig.update_yaxes(tickvals=[0,1,2,3,4], ticktext=['None','Alert','Recal','Partial','Full'], row=3, col=1)

        fig.update_layout(height=700, template="plotly_dark",
                         legend=dict(orientation="h", y=1.03, x=0.5, xanchor="center"),
                         margin=dict(l=50, r=30, t=70, b=30))
        st.plotly_chart(fig, use_container_width=True)

        # Phase summary table
        st.markdown("#### Phase Summary")
        phase_summary = []
        for phase in sorted(fdf['true_phase'].unique()):
            pdata_ph = fdf[fdf['true_phase'] == phase]
            phase_summary.append({
                'Phase': phase, 'Windows': len(pdata_ph),
                'D_total (mean)': round(pdata_ph['D_total'].mean(), 4),
                'D_total (max)': round(pdata_ph['D_total'].max(), 4),
                'D_SHAP (mean)': round(pdata_ph['D_shap'].mean(), 4),
                'Adv Score (mean)': round(pdata_ph['adv_score'].mean(), 4),
                'Risk (mean)': round(pdata_ph['risk_score'].mean(), 4),
                'Est AUPRC (mean)': round(pdata_ph['est_auprc'].mean(), 4),
                'Max Intervention': int(pdata_ph['intervention_level'].max()),
            })
        st.dataframe(pd.DataFrame(phase_summary), use_container_width=True)

    # ── Tab: Per-Window Data ──────────────────────────────────────────────
    with tabs[1]:
        st.subheader("Per-Window Detailed Data")
        st.markdown("*Raw pipeline output — every window's metrics, flags, and actions*")

        # Column selector
        all_cols = list(fdf.columns)
        numeric_cols = [c for c in all_cols if fdf[c].dtype in ['float64', 'int64', 'bool']]
        default_show = ['window_num', 'true_phase', 'D_total', 'D_shap', 'adv_score',
                       'drift_belief', 'risk_score', 'est_auprc', 'intervention_level',
                       'intervention_action', 'dpd', 'eod']
        show_cols = st.multiselect("Columns to display", all_cols,
                                  default=[c for c in default_show if c in all_cols], key='pw_cols')

        if show_cols:
            styled_df = fdf[show_cols].copy()
            st.dataframe(styled_df, use_container_width=True, height=500)

            # Statistics
            st.markdown("#### Descriptive Statistics")
            stat_cols = [c for c in show_cols if c in numeric_cols]
            if stat_cols:
                st.dataframe(fdf[stat_cols].describe().round(4), use_container_width=True)

        # Download
        csv_data = fdf.to_csv(index=False)
        st.download_button("📥 Download filtered data as CSV", csv_data,
                          "drift2act_filtered.csv", "text/csv", key='dl_csv')

    # ── Tab: SADI Components ─────────────────────────────────────────────
    with tabs[2]:
        st.subheader("SADI Component Breakdown")

        # D_total decomposition
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fdf['window_num'], y=fdf['D_total'], mode='lines',
                                name='D_total', line=dict(color='#3498db', width=3)))
        fig.add_trace(go.Scatter(x=fdf['window_num'], y=fdf['D_shap'], mode='lines',
                                name='D_SHAP (α=0.5)', line=dict(color='#9b59b6', width=2)))
        fig.add_trace(go.Scatter(x=fdf['window_num'], y=fdf['drift_belief'], mode='lines',
                                name='Drift Belief', line=dict(color='#2ecc71', width=1.5, dash='dot')))
        fig.add_trace(go.Scatter(x=fdf['window_num'], y=fdf['risk_score'], mode='lines',
                                name='Risk Score', line=dict(color='#e74c3c', width=1.5, dash='dash')))
        add_phase_shading(fig, fdf)
        fig.add_hline(y=thresh, line_dash="dash", line_color="#e74c3c", opacity=0.5)
        fig.update_layout(height=450, template="plotly_dark", xaxis_title="Window", yaxis_title="Score",
                         legend=dict(orientation="h", y=1.05, x=0.5, xanchor="center"))
        st.plotly_chart(fig, use_container_width=True)

        # Correlation matrix
        st.markdown("#### Component Correlations")
        corr_cols = ['D_total', 'D_shap', 'adv_score', 'drift_belief', 'risk_score', 'est_auprc']
        corr_cols = [c for c in corr_cols if c in fdf.columns]
        corr = fdf[corr_cols].corr().round(3)
        fig = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.columns,
            colorscale='RdBu_r', zmid=0, text=corr.values, texttemplate="%{text}",
            colorbar=dict(title="Corr", len=0.6)))
        fig.update_layout(height=400, template="plotly_dark", margin=dict(l=100))
        st.plotly_chart(fig, use_container_width=True)

        # SADI formula reference
        with st.expander("📚 SADI Formula Reference"):
            st.markdown("""
            **Per-feature SADI:**
            ```
            SADI(f, t) = α · KL(S_{t-1}(f) || S_t(f)) + β · |rank_t(f) - rank_{t-1}(f)| / N + γ · 𝟙[sign(μ_{t-1}(f)) ≠ sign(μ_t(f))]
            ```
            Default weights: `α=0.5, β=0.3, γ=0.2`

            **Overall drift score:**
            ```
            D_total = α · D_SHAP + β · D_feature + γ · D_confidence
            ```
            Where:
            - `D_SHAP` = mean SADI of top-10 features
            - `D_feature` = mean PSI across features
            - `D_confidence` = Wasserstein distance of prediction distributions
            """)

    # ── Tab: Feature Attribution ──────────────────────────────────────────
    with tabs[3]:
        st.subheader("Feature-Level Attribution")

        feature_data = []
        for _, row in fdf.iterrows():
            feats = row.get('top_sadi_features', '[]')
            if isinstance(feats, str):
                try: feats = json.loads(feats)
                except: feats = []
            for rank, feat in enumerate(feats):
                feature_data.append({'window': row['window_num'], 'feature': feat,
                                    'rank': rank + 1, 'phase': row['true_phase']})

        if feature_data:
            feat_df = pd.DataFrame(feature_data)
            freq = feat_df['feature'].value_counts()

            c1, c2 = st.columns(2)
            with c1:
                fig = go.Figure(go.Bar(x=freq.values[:15], y=freq.index[:15], orientation='h',
                                     marker=dict(color=freq.values[:15], colorscale='YlOrRd', showscale=True),
                                     text=freq.values[:15], textposition='outside'))
                fig.update_layout(title="Feature Flag Frequency", height=450, template="plotly_dark",
                                yaxis=dict(autorange="reversed"), margin=dict(l=160, r=60))
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                top = freq.head(12).index.tolist()
                hm = feat_df[feat_df['feature'].isin(top)].pivot_table(
                    index='feature', columns='window', values='rank', aggfunc='min', fill_value=0).reindex(top)
                fig = go.Figure(go.Heatmap(
                    z=hm.values, x=[f"W{c}" for c in hm.columns], y=hm.index,
                    colorscale=[[0,'#1a1a2e'],[0.2,'#2ecc71'],[0.5,'#f1c40f'],[1,'#e74c3c']],
                    text=hm.values, texttemplate="%{text}"))
                fig.update_layout(title="Feature Rank Across Windows", height=450, template="plotly_dark",
                                margin=dict(l=160), xaxis=dict(tickangle=-45))
                st.plotly_chart(fig, use_container_width=True)

            # KS / PSI flags
            st.markdown("#### Statistical Test Flags")
            ks_data = []
            for _, row in fdf.iterrows():
                ks_f = row.get('ks_flagged_features', '[]')
                psi_f = row.get('psi_flagged_features', '[]')
                if isinstance(ks_f, str):
                    try: ks_f = json.loads(ks_f)
                    except: ks_f = []
                if isinstance(psi_f, str):
                    try: psi_f = json.loads(psi_f)
                    except: psi_f = []
                ks_data.append({'window': row['window_num'], 'KS flagged': len(ks_f),
                               'PSI flagged': len(psi_f), 'KS detected': row.get('ks_detected', False),
                               'PSI detected': row.get('psi_detected', False)})
            st.dataframe(pd.DataFrame(ks_data), use_container_width=True, height=300)

            # Ground truth comparison
            if drift_gt:
                st.markdown("#### 🎯 Ground Truth vs Detection")
                gt_f = set(drift_gt.keys())
                det_f = set(freq.index)
                tp = gt_f & det_f; fp = det_f - gt_f; fn = gt_f - det_f
                prec = len(tp) / max(len(tp) + len(fp), 1)
                rec = len(tp) / max(len(tp) + len(fn), 1)
                f1 = 2 * prec * rec / max(prec + rec, 1e-6)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("TP", len(tp)); c2.metric("FP", len(fp))
                c3.metric("FN", len(fn)); c4.metric("F1", f"{f1:.3f}")
                if tp: st.success(f"Detected: {', '.join(sorted(tp))}")
                if fn: st.warning(f"Missed: {', '.join(sorted(fn))}")
                if fp: st.info(f"Extra flags: {', '.join(sorted(fp))}")

    # ── Tab: Model Diagnostics ───────────────────────────────────────────
    with tabs[4]:
        st.subheader("Model Performance Diagnostics")

        if baseline:
            st.markdown("#### Baseline Model Metrics")
            models_data = []
            for mname, mdata in baseline.items():
                if isinstance(mdata, dict):
                    models_data.append({'Model': mname, **{k: round(v, 4) if isinstance(v, float) else v
                                                           for k, v in mdata.items()}})
            if models_data:
                st.dataframe(pd.DataFrame(models_data), use_container_width=True)

        # AUPRC degradation
        st.markdown("#### Performance Degradation Over Time")
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Est. AUPRC Over Windows", "Risk Score Over Windows"))
        add_phase_shading(fig, fdf, 1, 1); add_phase_shading(fig, fdf, 1, 2)
        fig.add_trace(go.Scatter(x=fdf['window_num'], y=fdf['est_auprc'], mode='lines+markers',
                                name='Est. AUPRC', line=dict(color='#3498db', width=2),
                                marker=dict(size=4)), row=1, col=1)
        if 'baseline_auprc' in fdf.columns:
            fig.add_hline(y=fdf['baseline_auprc'].iloc[0], line_dash="dash", line_color="#2ecc71",
                         annotation_text="Baseline", row=1, col=1)
        fig.add_trace(go.Scatter(x=fdf['window_num'], y=fdf['risk_score'], mode='lines+markers',
                                name='Risk Score', line=dict(color='#e74c3c', width=2),
                                marker=dict(size=4)), row=1, col=2)
        fig.update_layout(height=380, template="plotly_dark", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # D_total vs AUPRC scatter
        fig = px.scatter(fdf, x='D_total', y='est_auprc', color='true_phase',
                        color_discrete_map={'phase1': '#2ecc71', 'phase2': '#f1c40f', 'phase3': '#e74c3c'},
                        size='risk_score', size_max=12, hover_data=['window_num', 'intervention_action'],
                        template='plotly_dark', height=380,
                        title="D_total vs Estimated AUPRC (size = risk score)")
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab: Detector Comparison ──────────────────────────────────────────
    with tabs[5]:
        st.subheader("Detector Comparison & Ablation")

        if t1 is not None:
            fig = go.Figure()
            for m, c in [('Precision', '#3498db'), ('Recall', '#2ecc71'), ('F1', '#e74c3c')]:
                if m in t1.columns:
                    fig.add_trace(go.Bar(name=m, x=t1['Detector'], y=t1[m], marker_color=c,
                                       text=t1[m].round(3), textposition='outside',
                                       textfont=dict(size=13, color='white')))
            fig.update_layout(barmode='group', height=420, template="plotly_dark",
                            title="Detection: Precision / Recall / F1", yaxis=dict(range=[0, 1.15]),
                            legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"))
            st.plotly_chart(fig, use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### Detection Metrics")
                st.dataframe(t1, use_container_width=True)
            with c2:
                if t2 is not None:
                    st.markdown("#### Attribution Accuracy")
                    st.dataframe(t2, use_container_width=True)

        if t3 is not None:
            st.markdown("---")
            st.markdown("#### 🔬 SADI Ablation Study")
            cfg_col = next((c for c in t3.columns if c.lower().startswith('config')), t3.columns[0])
            f1_col = 'F1' if 'F1' in t3.columns else t3.columns[-1]
            n = len(t3)
            colors = ['#95a5a6'] * (n - 1) + ['#3498db']
            fig = go.Figure(go.Bar(x=t3[cfg_col], y=t3[f1_col], marker=dict(color=colors),
                                  text=t3[f1_col].round(3), textposition='outside'))
            fig.update_layout(yaxis=dict(range=[0, 1.1], title="F1"), height=380,
                            template="plotly_dark", title="Ablation: F1 by SADI Config")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(t3, use_container_width=True)

        if stat_sig:
            st.markdown("---")
            st.markdown("#### 📈 Statistical Significance (Bootstrap)")
            c1, c2, c3 = st.columns(3)
            c1.metric("SADI F1", f"{stat_sig.get('sadi_f1_mean', 0):.3f} ± {stat_sig.get('sadi_f1_std', 0):.3f}")
            c2.metric("KS F1", f"{stat_sig.get('ks_f1_mean', 0):.3f} ± {stat_sig.get('ks_f1_std', 0):.3f}")
            p = stat_sig.get('wilcoxon_p_value', 1)
            c3.metric("Wilcoxon p", f"{p:.6f}",
                     delta="✓ Significant" if stat_sig.get('significant') else "Not significant")

    # ── Tab: Fairness Audit ───────────────────────────────────────────────
    with tabs[6]:
        st.subheader("Fairness Audit")

        fig = make_subplots(rows=1, cols=2, subplot_titles=("Demographic Parity Diff", "Equalized Odds Diff"))
        add_phase_shading(fig, fdf, 1, 1); add_phase_shading(fig, fdf, 1, 2)
        fig.add_trace(go.Scatter(x=fdf['window_num'], y=fdf['dpd'], mode='lines+markers',
                                name='DPD', line=dict(color='#e74c3c', width=2), marker=dict(size=5)), row=1, col=1)
        fig.add_hline(y=0.10, line_dash="dash", line_color="#f39c12",
                      annotation_text="Threshold 0.10", row=1, col=1)
        fig.add_trace(go.Scatter(x=fdf['window_num'], y=fdf['eod'], mode='lines+markers',
                                name='EOD', line=dict(color='#3498db', width=2), marker=dict(size=5)), row=1, col=2)
        fig.add_hline(y=0.10, line_dash="dash", line_color="#f39c12",
                      annotation_text="Threshold 0.10", row=1, col=2)
        fig.update_layout(height=400, template="plotly_dark", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mean DPD", f"{fdf['dpd'].mean():.4f}")
        c2.metric("Max DPD", f"{fdf['dpd'].max():.4f}")
        c3.metric("Mean EOD", f"{fdf['eod'].mean():.4f}")
        c4.metric("Max EOD", f"{fdf['eod'].max():.4f}")

        if t4 is not None:
            st.markdown("#### Phase-Level Fairness")
            st.dataframe(t4, use_container_width=True)

        # Fairness vs drift correlation
        st.markdown("#### Fairness-Drift Correlation")
        fig = px.scatter(fdf, x='D_total', y='dpd', color='true_phase',
                        color_discrete_map={'phase1': '#2ecc71', 'phase2': '#f1c40f', 'phase3': '#e74c3c'},
                        template='plotly_dark', height=300,
                        title="Does drift cause fairness degradation?")
        fig.add_hline(y=0.10, line_dash="dash", line_color="#f39c12")
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab: Alert Timeline ──────────────────────────────────────────────
    with tabs[7]:
        st.subheader("Alert Timeline")

        if alerts:
            st.markdown(f"**{len(alerts)} total alerts**")

            # Alert level distribution
            alert_levels = [a.get('level', 0) for a in alerts]
            al_df = pd.Series(alert_levels).value_counts().sort_index()
            level_map = {1: 'L1 Alert', 2: 'L2 Recalibrate', 3: 'L3 Partial Retrain', 4: 'L4 Full Retrain'}
            fig = go.Figure(go.Bar(
                x=[level_map.get(int(i), f'L{i}') for i in al_df.index],
                y=al_df.values,
                marker_color=[lc.get(int(i), '#888') for i in al_df.index]))
            fig.update_layout(height=250, template="plotly_dark", title="Alert Distribution",
                            margin=dict(t=50, b=30))
            st.plotly_chart(fig, use_container_width=True)

            # Individual alerts
            sev_filter = st.selectbox("Min severity", [1, 2, 3, 4],
                                     format_func=lambda x: f"L{x} — {level_map.get(x, '?')}", key='p_sev')
            fa = [a for a in alerts if a.get('level', 0) >= sev_filter]
            st.markdown(f"Showing **{len(fa)}** of {len(alerts)}")

            for a in fa[-20:]:
                lv_a = a.get('level', 0)
                ic = {1: '🔵', 2: '🟡', 3: '🟠', 4: '🔴'}.get(lv_a, '⚪')
                label = f"{ic} W{a.get('window_idx', '?')} — {a.get('action', 'N/A')} ({a.get('phase', '?')})"
                with st.expander(label):
                    if 'alert_text' in a:
                        st.markdown(f'<div class="alert-box">{a["alert_text"]}</div>', unsafe_allow_html=True)
                    else:
                        st.json(a)
        else:
            st.markdown('<div style="text-align:center;padding:3rem;color:#888">'
                       '<div style="font-size:3rem">✅</div>'
                       '<div style="font-size:1.2rem;font-weight:600">No alerts in this pipeline run</div></div>',
                       unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 🏥 Drift2Act")
    mode = st.radio(
        "Dashboard Mode",
        ["📤 Upload & Analyze", "🔧 Pipeline Analysis", "📊 Review Pipeline Results"],
        help="Upload: test new patient data. Pipeline: upload full pipeline results. Review: browse local results.",
        captions=["For clinical data teams", "For MLOps engineers", "For local pipeline runs"]
    )
    st.markdown("---")

if mode == "📤 Upload & Analyze":
    render_upload_mode()
elif mode == "🔧 Pipeline Analysis":
    render_pipeline_upload_mode()
else:
    render_review_mode()

# Footer
st.markdown(
    "<hr style='border:none;border-top:1px solid rgba(255,255,255,0.06);margin:1.5rem 0'>"
    "<div style='text-align:center;color:#555;font-size:0.82rem;padding:0.5rem 0 1rem'>"
    "Drift2Act v1.0 · Explainable, Proactive Concept Drift Detection for Clinical Sepsis Prediction<br>"
    "IEM-UEM Kolkata · Department of CSE (AI) · 4A-FINAL</div>",
    unsafe_allow_html=True
)
