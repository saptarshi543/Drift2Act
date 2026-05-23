"""
Drift2Act — Streamlit Dashboard
================================

Interactive dashboard for monitoring concept drift in clinical sepsis prediction.

Panels:
1. Header KPIs: Current D_total, alert status, intervention level, estimated AUPRC
2. SADI Timeline: Line plot with threshold and phase annotations
3. Feature Drift Heatmap: Top features × windows colour-coded by SADI
4. SHAP Importance Comparison: Phase 1 vs current
5. Drift2Act Intervention Log: Table of all actions
6. Fairness Panel: DPD and EOD over time
7. Detector Comparison: F1 comparison across methods
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path

# ── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Drift2Act — Clinical Drift Monitor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        padding: 1.8rem 2.2rem;
        border-radius: 14px;
        margin-bottom: 1.5rem;
        color: white;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }

    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }

    .main-header p {
        color: #b0b0e0;
        margin: 0.4rem 0 0 0;
        font-size: 0.95rem;
        font-weight: 300;
    }

    .kpi-row {
        display: flex;
        gap: 1rem;
        margin-bottom: 1.5rem;
    }

    .kpi-card {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 1.1rem 1.4rem;
        text-align: center;
        color: white;
        flex: 1;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }

    .kpi-card .kpi-value {
        font-size: 1.9rem;
        font-weight: 700;
        margin: 0.2rem 0;
        line-height: 1.2;
    }

    .kpi-card .kpi-label {
        font-size: 0.72rem;
        color: #8888aa;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 500;
    }

    .kpi-card .kpi-delta {
        font-size: 0.78rem;
        margin-top: 0.2rem;
        font-weight: 500;
    }

    .severity-low { color: #2ecc71; }
    .severity-moderate { color: #f1c40f; }
    .severity-high { color: #e67e22; }
    .severity-critical { color: #e74c3c; }

    .alert-box {
        background: #0d1117;
        border-left: 4px solid #e74c3c;
        padding: 1rem 1.2rem;
        border-radius: 6px;
        font-family: 'JetBrains Mono', 'Courier New', monospace;
        font-size: 0.82rem;
        white-space: pre-wrap;
        color: #c9d1d9;
        max-height: 400px;
        overflow-y: auto;
        line-height: 1.5;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 0.9rem 1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }

    div[data-testid="stMetric"] label {
        font-size: 0.75rem !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: rgba(26,26,46,0.5);
        border-radius: 10px;
        padding: 4px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.5rem 1.1rem;
        font-weight: 500;
        font-size: 0.88rem;
    }

    .section-divider {
        border: none;
        border-top: 1px solid rgba(255,255,255,0.06);
        margin: 1.5rem 0;
    }

    /* Scrollbar styling */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0d1117; }
    ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Data Loading ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@st.cache_data
def load_results():
    """Load pipeline results."""
    results_path = PROJECT_ROOT / 'results' / 'drift2act_results.csv'
    if not results_path.exists():
        return None
    return pd.read_csv(results_path)


@st.cache_data
def load_baseline_metrics():
    """Load baseline model metrics."""
    path = PROJECT_ROOT / 'results' / 'baseline_metrics.json'
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_alerts():
    """Load alert log."""
    path = PROJECT_ROOT / 'results' / 'alert_log.json'
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_comparison_table():
    """Load detector comparison table."""
    path = PROJECT_ROOT / 'results' / 'tables' / 'table1_detector_comparison.csv'
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_ablation_table():
    """Load ablation study table."""
    path = PROJECT_ROOT / 'results' / 'tables' / 'table3_ablation.csv'
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_fairness_table():
    """Load fairness metrics table."""
    path = PROJECT_ROOT / 'results' / 'tables' / 'table4_fairness.csv'
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_attribution_table():
    """Load attribution accuracy table."""
    path = PROJECT_ROOT / 'results' / 'tables' / 'table2_attribution_accuracy.csv'
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_drift_ground_truth():
    """Load drift ground truth."""
    path = PROJECT_ROOT / 'results' / 'drift_ground_truth.json'
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_statistical_significance():
    """Load statistical significance results."""
    path = PROJECT_ROOT / 'results' / 'statistical_significance.json'
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


# ── Load Data ───────────────────────────────────────────────────────────────
results_df = load_results()
baseline_metrics = load_baseline_metrics()
alerts = load_alerts()
comparison_table = load_comparison_table()
ablation_table = load_ablation_table()
fairness_table = load_fairness_table()
attribution_table = load_attribution_table()
drift_gt = load_drift_ground_truth()
stat_sig = load_statistical_significance()

if results_df is None:
    st.markdown("""
    <div class="main-header">
        <h1>🏥 Drift2Act</h1>
        <p>Explainable, Proactive Concept Drift Detection for Clinical Sepsis Prediction</p>
    </div>
    """, unsafe_allow_html=True)

    st.error("⚠️ No results found. Please run the pipeline first:")
    st.code("python scripts/run_pipeline.py", language="bash")
    st.info("Place PhysioNet 2019 `.psv` files in `data/raw/training/` before running, or the pipeline will use synthetic fallback data.")
    st.stop()

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Dashboard Controls")

    sadi_threshold = st.slider(
        "SADI Drift Threshold",
        min_value=0.1, max_value=2.0, value=0.30, step=0.05,
        help="D_total threshold for drift detection"
    )

    window_range = st.slider(
        "Window Range",
        min_value=0,
        max_value=int(results_df['window_num'].max()),
        value=(0, int(results_df['window_num'].max())),
        help="Filter windows to display"
    )

    phase_filter = st.multiselect(
        "Phase Filter",
        options=['phase1', 'phase2', 'phase3'],
        default=['phase1', 'phase2', 'phase3']
    )

    st.markdown("---")

    # Dataset summary
    st.markdown("### 📋 Run Summary")
    phase_counts = results_df['true_phase'].value_counts()
    for phase, count in sorted(phase_counts.items()):
        emoji = {'phase1': '🟢', 'phase2': '🟡', 'phase3': '🔴'}.get(phase, '⚪')
        st.markdown(f"{emoji} **{phase}**: {count} windows")

    st.markdown(f"📊 **Total windows:** {len(results_df)}")

    if drift_gt:
        st.markdown(f"🎯 **Drifted columns:** {len(drift_gt)}")

    # Model performance summary
    if baseline_metrics:
        st.markdown("---")
        st.markdown("### 🧠 Model Performance")
        xgb = baseline_metrics.get('xgboost', baseline_metrics.get('xgboost_calibrated', {}))
        lr = baseline_metrics.get('logistic_regression', {})
        st.markdown(f"**XGBoost AUPRC:** {xgb.get('auprc', 0):.4f}")
        st.markdown(f"**XGBoost AUROC:** {xgb.get('auroc', 0):.4f}")
        st.markdown(f"**LR AUPRC:** {lr.get('auprc', 0):.4f}")

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;color:#666;font-size:0.78rem;'>"
        "Drift2Act v1.0<br>IEM-UEM Kolkata, CSE (AI)"
        "</div>",
        unsafe_allow_html=True
    )

# ── Filter Data ─────────────────────────────────────────────────────────────
mask = (
    (results_df['window_num'] >= window_range[0]) &
    (results_df['window_num'] <= window_range[1]) &
    (results_df['true_phase'].isin(phase_filter))
)
filtered_df = results_df[mask].copy()

# ═══════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="main-header">
    <h1>🏥 Drift2Act</h1>
    <p>Explainable, Proactive Concept Drift Detection for Clinical Sepsis Prediction</p>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# KPI CARDS
# ═══════════════════════════════════════════════════════════════════════════

if len(filtered_df) > 0:
    latest = filtered_df.iloc[-1]

    d_total = latest['D_total']
    if d_total < 0.3:
        severity, sev_color = 'NOMINAL', '#2ecc71'
    elif d_total < 0.6:
        severity, sev_color = 'MODERATE', '#f1c40f'
    elif d_total < 1.0:
        severity, sev_color = 'HIGH', '#e67e22'
    else:
        severity, sev_color = 'CRITICAL', '#e74c3c'

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
            <div class="kpi-label">Intervention Level</div>
            <div class="kpi-value" style="color:{level_colors[level]}">L{level}</div>
            <div class="kpi-delta" style="color:{level_colors[level]}">{level_names[level]}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Est. AUPRC</div>
            <div class="kpi-value">{est_auprc:.3f}</div>
            <div class="kpi-delta" style="color:{'#e74c3c' if perf_drop > 0.02 else '#2ecc71'}">{'▼' if perf_drop > 0 else '▲'} {perf_drop:.3f} from baseline</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Risk Score</div>
            <div class="kpi-value">{latest['risk_score']:.3f}</div>
            <div class="kpi-delta" style="color:#8888aa">Perf. degradation est.</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Drift Belief</div>
            <div class="kpi-value">{latest.get('drift_belief', 0):.3f}</div>
            <div class="kpi-delta" style="color:#8888aa">SADI + Adversarial</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# MAIN TABS
# ═══════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📈 SADI Timeline",
    "🔥 Feature Drift",
    "🧬 SHAP Analysis",
    "🎯 Interventions",
    "⚖️ Fairness",
    "📊 Detectors",
    "🚨 Alerts"
])

PHASE_COLORS = {
    'phase1': 'rgba(46,204,113,0.12)',
    'phase2': 'rgba(241,196,15,0.12)',
    'phase3': 'rgba(231,76,60,0.12)'
}
PHASE_LINE_COLORS = {
    'phase1': '#2ecc71',
    'phase2': '#f1c40f',
    'phase3': '#e74c3c'
}

# Helper to add phase shading to any figure
def add_phase_shading(fig, df, row=1, col=1):
    for phase, color in PHASE_COLORS.items():
        phase_data = df[df['true_phase'] == phase]
        if len(phase_data) > 0:
            fig.add_vrect(
                x0=phase_data['window_num'].min() - 0.5,
                x1=phase_data['window_num'].max() + 0.5,
                fillcolor=color, layer="below", line_width=0,
                annotation_text=phase.replace('phase', 'P'),
                annotation_position="top left",
                annotation_font=dict(size=10, color=PHASE_LINE_COLORS[phase]),
                row=row, col=col
            )


# ── Tab 1: SADI Timeline ───────────────────────────────────────────────────
with tab1:
    st.subheader("SADI Drift Score Over Time")

    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.5, 0.25, 0.25],
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=("D_total Composite Score", "Component Breakdown", "Adversarial Drift Score")
    )

    add_phase_shading(fig, filtered_df, row=1, col=1)
    add_phase_shading(fig, filtered_df, row=2, col=1)
    add_phase_shading(fig, filtered_df, row=3, col=1)

    # D_total line
    fig.add_trace(
        go.Scatter(
            x=filtered_df['window_num'], y=filtered_df['D_total'],
            mode='lines', name='D_total',
            line=dict(color='#3498db', width=3),
            fill='tozeroy', fillcolor='rgba(52,152,219,0.08)'
        ), row=1, col=1
    )

    # Threshold line
    fig.add_hline(y=sadi_threshold, line_dash="dash", line_color="#e74c3c",
                  annotation_text=f"Threshold ({sadi_threshold})",
                  annotation_font=dict(color="#e74c3c", size=10),
                  row=1, col=1)

    # Component breakdown (row 2)
    fig.add_trace(
        go.Scatter(
            x=filtered_df['window_num'], y=filtered_df['D_shap'],
            mode='lines', name='D_SHAP',
            line=dict(color='#9b59b6', width=1.5)
        ), row=2, col=1
    )

    # Adversarial score (row 3)
    fig.add_trace(
        go.Scatter(
            x=filtered_df['window_num'], y=filtered_df['adv_score'],
            mode='lines+markers', name='Adversarial AUROC',
            line=dict(color='#e67e22', width=2),
            marker=dict(size=4)
        ), row=3, col=1
    )
    fig.add_hline(y=0.5, line_dash="dash", line_color="#555",
                  annotation_text="No drift (0.5)",
                  annotation_font=dict(color="#888", size=9),
                  row=3, col=1)

    fig.update_layout(
        height=650, template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.03, x=0.5, xanchor="center"),
        margin=dict(l=50, r=30, t=70, b=30),
        font=dict(family="Inter"),
    )
    fig.update_yaxes(title_text="D_total", row=1, col=1)
    fig.update_yaxes(title_text="Score", row=2, col=1)
    fig.update_yaxes(title_text="AUROC", row=3, col=1)
    fig.update_xaxes(title_text="Window Index", row=3, col=1)

    st.plotly_chart(fig, use_container_width=True)

    # Summary stats row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mean D_total", f"{filtered_df['D_total'].mean():.3f}")
    col2.metric("Max D_total", f"{filtered_df['D_total'].max():.3f}")
    col3.metric("Windows > Threshold", f"{(filtered_df['D_total'] > sadi_threshold).sum()}/{len(filtered_df)}")
    col4.metric("Mean Adv. Score", f"{filtered_df['adv_score'].mean():.3f}")


# ── Tab 2: Feature Drift ───────────────────────────────────────────────────
with tab2:
    st.subheader("Feature-Level Drift Analysis")

    try:
        # Collect per-window top features
        feature_data = []
        for _, row in filtered_df.iterrows():
            feats = row.get('top_sadi_features', '[]')
            if isinstance(feats, str):
                try:
                    feats = json.loads(feats)
                except Exception:
                    feats = []
            for rank, feat in enumerate(feats):
                feature_data.append({
                    'window': row['window_num'],
                    'feature': feat,
                    'rank': rank + 1,
                    'phase': row['true_phase']
                })

        if feature_data:
            feat_df = pd.DataFrame(feature_data)

            col1, col2 = st.columns([1, 1])

            with col1:
                # Most flagged features bar chart
                feature_freq = feat_df['feature'].value_counts().head(15)
                fig = go.Figure(go.Bar(
                    x=feature_freq.values,
                    y=feature_freq.index,
                    orientation='h',
                    marker=dict(
                        color=feature_freq.values,
                        colorscale='YlOrRd',
                        showscale=True,
                        colorbar=dict(title="Count", len=0.6)
                    ),
                    text=feature_freq.values,
                    textposition='outside'
                ))
                fig.update_layout(
                    title="Most Frequently Flagged Features",
                    xaxis_title="Times in Top-5 SADI",
                    height=450, template="plotly_dark",
                    yaxis=dict(autorange="reversed"),
                    margin=dict(l=160, r=50)
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Feature × window heatmap
                top_feats = feature_freq.head(10).index.tolist()
                heatmap_data = feat_df[feat_df['feature'].isin(top_feats)].copy()

                if len(heatmap_data) > 0:
                    pivot = heatmap_data.pivot_table(
                        index='feature', columns='window', values='rank',
                        aggfunc='min', fill_value=0
                    )
                    # Reorder rows by frequency
                    pivot = pivot.reindex(top_feats)

                    fig = go.Figure(go.Heatmap(
                        z=pivot.values,
                        x=[f"W{c}" for c in pivot.columns],
                        y=pivot.index,
                        colorscale=[[0, '#1a1a2e'], [0.2, '#2ecc71'], [0.5, '#f1c40f'], [1, '#e74c3c']],
                        colorbar=dict(title="Rank", len=0.6),
                        text=pivot.values,
                        texttemplate="%{text}",
                        hovertemplate="Feature: %{y}<br>Window: %{x}<br>SADI Rank: %{z}<extra></extra>"
                    ))
                    fig.update_layout(
                        title="SADI Feature Rank Heatmap",
                        height=450, template="plotly_dark",
                        margin=dict(l=160),
                        xaxis=dict(tickangle=-45)
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # Drift ground truth comparison
            if drift_gt:
                st.markdown("---")
                st.markdown("#### 🎯 Drift Ground Truth vs SADI Detection")

                gt_features = set(drift_gt.keys())
                detected_features = set(feature_freq.index)
                tp = gt_features & detected_features
                fp = detected_features - gt_features
                fn = gt_features - detected_features

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("✅ True Positives", len(tp))
                col2.metric("⚠️ False Positives", len(fp))
                col3.metric("❌ Missed", len(fn))
                precision = len(tp) / max(len(tp) + len(fp), 1)
                recall = len(tp) / max(len(tp) + len(fn), 1)
                col4.metric("F1 Score", f"{2*precision*recall/max(precision+recall, 1e-6):.3f}")

                if tp:
                    st.success(f"Correctly detected: {', '.join(sorted(tp))}")
                if fn:
                    st.warning(f"Missed drifted features: {', '.join(sorted(fn))}")
        else:
            st.info("No feature-level SADI data available in the results.")
    except Exception as e:
        st.error(f"Error parsing feature data: {e}")


# ── Tab 3: SHAP Analysis ──────────────────────────────────────────────────
with tab3:
    st.subheader("SHAP Feature Importance")

    col1, col2 = st.columns(2)

    with col1:
        beeswarm_path = PROJECT_ROOT / 'paper' / 'figures' / 'fig1_shap_beeswarm.png'
        if beeswarm_path.exists():
            st.image(str(beeswarm_path), caption="SHAP Beeswarm — Phase 1 Baseline")
        else:
            st.info("SHAP beeswarm not yet generated.")

    with col2:
        dist_path = PROJECT_ROOT / 'paper' / 'figures' / 'fig1_feature_distributions.png'
        if dist_path.exists():
            st.image(str(dist_path), caption="Feature Distributions Across Phases")
        else:
            st.info("Feature distribution plot not yet generated.")

    # Attribution accuracy table
    if attribution_table is not None:
        st.markdown("---")
        st.markdown("#### 📐 Feature Attribution Accuracy")
        st.markdown("*How well each detector identifies which specific features drifted*")

        fig = go.Figure()
        for i, metric in enumerate(['Precision', 'Recall']):
            if metric in attribution_table.columns:
                fig.add_trace(go.Bar(
                    name=metric,
                    x=attribution_table['Detector'],
                    y=attribution_table[metric],
                    marker_color=['#3498db', '#2ecc71'][i],
                    text=attribution_table[metric].round(3),
                    textposition='outside'
                ))
        fig.update_layout(
            title="Feature Attribution: Which Detector Best Identifies Drifted Features?",
            barmode='group',
            height=350, template="plotly_dark",
            yaxis=dict(range=[0, 1.1])
        )
        st.plotly_chart(fig, use_container_width=True)


# ── Tab 4: Interventions ───────────────────────────────────────────────────
with tab4:
    st.subheader("Drift2Act Intervention Log")

    level_labels = {0: 'No Action', 1: 'Alert', 2: 'Recalibrate', 3: 'Partial Retrain', 4: 'Full Retrain'}
    level_colors_list = ['#2ecc71', '#3498db', '#f1c40f', '#e67e22', '#e74c3c']

    col1, col2 = st.columns([1, 2])

    with col1:
        level_counts = filtered_df['intervention_level'].value_counts().sort_index()
        fig = go.Figure(go.Pie(
            labels=[level_labels.get(int(i), str(i)) for i in level_counts.index],
            values=level_counts.values,
            marker=dict(colors=[level_colors_list[int(i)] for i in level_counts.index]),
            hole=0.55,
            textinfo='label+value',
            textfont=dict(size=12)
        ))
        fig.update_layout(
            title="Intervention Distribution",
            height=350, template="plotly_dark",
            margin=dict(l=20, r=20, t=50, b=20),
            showlegend=False,
            annotations=[dict(text=f"{len(filtered_df)}<br>windows", x=0.5, y=0.5,
                            font_size=16, showarrow=False, font_color="white")]
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Intervention timeline with phase coloring
        fig = go.Figure()

        for phase, pcolor in PHASE_LINE_COLORS.items():
            phase_data = filtered_df[filtered_df['true_phase'] == phase]
            if len(phase_data) > 0:
                fig.add_trace(go.Scatter(
                    x=phase_data['window_num'],
                    y=phase_data['intervention_level'],
                    mode='markers+lines',
                    name=phase.replace('phase', 'Phase '),
                    marker=dict(
                        color=[level_colors_list[int(lv)] for lv in phase_data['intervention_level']],
                        size=8, line=dict(width=1, color='white')
                    ),
                    line=dict(color=pcolor, width=1.5, dash='dot')
                ))

        fig.update_layout(
            title="Intervention Level Over Time",
            xaxis_title="Window", yaxis_title="Level",
            yaxis=dict(tickvals=[0, 1, 2, 3, 4],
                       ticktext=['None', 'Alert', 'Recalib.', 'Partial', 'Full']),
            height=350, template="plotly_dark",
            margin=dict(l=60, r=30, t=50, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        st.plotly_chart(fig, use_container_width=True)

    # D_total vs Est. AUPRC scatter
    st.markdown("#### Risk Analysis: D_total vs Performance")
    fig = px.scatter(
        filtered_df, x='D_total', y='est_auprc',
        color='true_phase',
        color_discrete_map={'phase1': '#2ecc71', 'phase2': '#f1c40f', 'phase3': '#e74c3c'},
        size='risk_score', size_max=12,
        hover_data=['window_num', 'intervention_action'],
        template='plotly_dark', height=350
    )
    fig.update_layout(
        xaxis_title="D_total (Drift Score)",
        yaxis_title="Estimated AUPRC",
        margin=dict(l=50, r=30, t=30, b=40)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Intervention detail table
    with st.expander("📋 Full Intervention Table", expanded=False):
        intervention_data = filtered_df[filtered_df['intervention_level'] >= 1][
            ['window_num', 'true_phase', 'D_total', 'D_shap', 'risk_score',
             'intervention_level', 'intervention_action', 'est_auprc']
        ].copy()
        intervention_data.columns = ['Window', 'Phase', 'D_total', 'D_SHAP', 'Risk',
                                     'Level', 'Action', 'Est. AUPRC']
        if len(intervention_data) > 0:
            st.dataframe(
                intervention_data.style.format({
                    'D_total': '{:.3f}', 'D_SHAP': '{:.3f}',
                    'Risk': '{:.4f}', 'Est. AUPRC': '{:.4f}'
                }),
                use_container_width=True, height=300
            )
        else:
            st.info("No interventions triggered in the selected range.")


# ── Tab 5: Fairness ────────────────────────────────────────────────────────
with tab5:
    st.subheader("Fairness Monitoring")
    st.markdown("*Tracking demographic parity and equalized odds across drift phases*")

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Demographic Parity Difference (DPD)", "Equalized Odds Difference (EOD)")
    )

    add_phase_shading(fig, filtered_df, row=1, col=1)
    add_phase_shading(fig, filtered_df, row=1, col=2)

    fig.add_trace(
        go.Scatter(
            x=filtered_df['window_num'], y=filtered_df['dpd'],
            mode='lines+markers', name='DPD',
            line=dict(color='#e74c3c', width=2),
            marker=dict(size=5)
        ), row=1, col=1
    )
    fig.add_hline(y=0.10, line_dash="dash", line_color="#f39c12",
                  annotation_text="Fairness bound (0.10)",
                  annotation_font=dict(size=9), row=1, col=1)

    fig.add_trace(
        go.Scatter(
            x=filtered_df['window_num'], y=filtered_df['eod'],
            mode='lines+markers', name='EOD',
            line=dict(color='#3498db', width=2),
            marker=dict(size=5)
        ), row=1, col=2
    )
    fig.add_hline(y=0.10, line_dash="dash", line_color="#f39c12",
                  annotation_text="Fairness bound (0.10)",
                  annotation_font=dict(size=9), row=1, col=2)

    fig.update_layout(
        height=400, template="plotly_dark",
        margin=dict(l=50, r=30, t=60, b=40),
        showlegend=False
    )
    fig.update_xaxes(title_text="Window", row=1, col=1)
    fig.update_xaxes(title_text="Window", row=1, col=2)
    fig.update_yaxes(title_text="DPD", row=1, col=1)
    fig.update_yaxes(title_text="EOD", row=1, col=2)
    st.plotly_chart(fig, use_container_width=True)

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mean DPD", f"{filtered_df['dpd'].mean():.4f}")
    col2.metric("Max DPD", f"{filtered_df['dpd'].max():.4f}")
    col3.metric("Mean EOD", f"{filtered_df['eod'].mean():.4f}")
    col4.metric("Max EOD", f"{filtered_df['eod'].max():.4f}")

    # Fairness table by phase
    if fairness_table is not None:
        with st.expander("📋 Fairness Metrics by Phase"):
            st.dataframe(fairness_table, use_container_width=True)


# ── Tab 6: Detector Comparison ──────────────────────────────────────────────
with tab6:
    st.subheader("Drift Detector Comparison")

    if comparison_table is not None:
        # Main comparison chart
        fig = go.Figure()
        bar_colors = {'Precision': '#3498db', 'Recall': '#2ecc71', 'F1': '#e74c3c'}
        for metric, color in bar_colors.items():
            if metric in comparison_table.columns:
                fig.add_trace(go.Bar(
                    name=metric,
                    x=comparison_table['Detector'],
                    y=comparison_table[metric],
                    marker_color=color,
                    text=comparison_table[metric].round(3),
                    textposition='outside',
                    textfont=dict(size=13, color='white')
                ))
        fig.update_layout(
            title="Drift Detection Performance (Precision / Recall / F1)",
            barmode='group',
            height=420, template="plotly_dark",
            yaxis=dict(range=[0, 1.15], title="Score"),
            margin=dict(l=50, r=30, t=60, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center")
        )
        st.plotly_chart(fig, use_container_width=True)

        # Data tables side by side
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Detection Metrics")
            st.dataframe(comparison_table, use_container_width=True)
        with col2:
            if attribution_table is not None:
                st.markdown("#### Attribution Accuracy")
                st.dataframe(attribution_table, use_container_width=True)
    else:
        st.info("Comparison table not yet generated. Run the pipeline first.")

    # Ablation study
    if ablation_table is not None:
        st.markdown("---")
        st.markdown("#### 🔬 SADI Ablation Study")
        st.markdown("*Testing the contribution of each SADI component*")

        fig = go.Figure(go.Bar(
            x=ablation_table['Configuration'] if 'Configuration' in ablation_table.columns
              else ablation_table.iloc[:, 0],
            y=ablation_table['F1'] if 'F1' in ablation_table.columns
              else ablation_table.iloc[:, -1],
            marker=dict(
                color=['#95a5a6', '#95a5a6', '#95a5a6', '#95a5a6', '#3498db'],
                line=dict(width=1.5, color='white')
            ),
            text=(ablation_table['F1'] if 'F1' in ablation_table.columns
                  else ablation_table.iloc[:, -1]).round(3),
            textposition='outside',
            textfont=dict(size=13)
        ))
        fig.update_layout(
            title="Ablation: F1 by SADI Component Configuration",
            yaxis=dict(range=[0, 1.1], title="F1 Score"),
            height=380, template="plotly_dark",
            margin=dict(l=50, r=30, t=60, b=80),
            xaxis=dict(tickangle=-20)
        )
        st.plotly_chart(fig, use_container_width=True)

    # Statistical significance
    if stat_sig:
        st.markdown("---")
        st.markdown("#### 📈 Statistical Significance (Bootstrap)")
        col1, col2, col3 = st.columns(3)
        col1.metric("SADI F1",
                     f"{stat_sig.get('sadi_f1_mean', 0):.3f} ± {stat_sig.get('sadi_f1_std', 0):.3f}")
        col2.metric("KS F1",
                     f"{stat_sig.get('ks_f1_mean', 0):.3f} ± {stat_sig.get('ks_f1_std', 0):.3f}")
        p_val = stat_sig.get('wilcoxon_p_value', 1)
        col3.metric("Wilcoxon p-value", f"{p_val:.4f}",
                     delta="✓ Significant" if stat_sig.get('significant', False) else "Not significant")


# ── Tab 7: Alerts ───────────────────────────────────────────────────────────
with tab7:
    st.subheader("Clinical Drift Alerts")

    if alerts and len(alerts) > 0:
        severity_filter = st.selectbox(
            "Minimum Alert Level",
            options=[1, 2, 3, 4],
            index=0,
            format_func=lambda x: {
                1: '🔵 L1 — Alert',
                2: '🟡 L2 — Recalibrate',
                3: '🟠 L3 — Partial Retrain',
                4: '🔴 L4 — Full Retrain'
            }[x]
        )

        filtered_alerts = [a for a in alerts if a.get('level', 0) >= severity_filter]
        st.markdown(f"**Showing {len(filtered_alerts)} of {len(alerts)} alerts**")

        for alert in filtered_alerts[-15:]:
            level = alert.get('level', 0)
            color = {1: '🔵', 2: '🟡', 3: '🟠', 4: '🔴'}.get(level, '⚪')
            action = alert.get('action', alert.get('intervention_action', 'N/A'))
            phase = alert.get('phase', alert.get('true_phase', '?'))

            with st.expander(
                f"{color} Window {alert.get('window_idx', alert.get('window_num', '?'))} "
                f"— {action} ({phase})"
            ):
                if 'alert_text' in alert:
                    st.markdown(
                        f'<div class="alert-box">{alert["alert_text"]}</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.json(alert)
    else:
        st.markdown("""
        <div style="text-align:center; padding:3rem; color:#888;">
            <div style="font-size:3rem; margin-bottom:1rem;">✅</div>
            <div style="font-size:1.2rem; font-weight:600;">No Critical Alerts</div>
            <div style="font-size:0.9rem; margin-top:0.5rem;">
                All intervention levels remained at L1 (Alert).<br>
                No L2+ actions were triggered during this run.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ── Footer ──────────────────────────────────────────────────────────────────
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.markdown(
    "<div style='text-align:center; color:#555; font-size:0.82rem; padding:0.5rem 0 1rem 0;'>"
    "Drift2Act v1.0 · Explainable, Proactive Concept Drift Detection for Clinical Sepsis Prediction<br>"
    "IEM-UEM Kolkata · Department of CSE (Artificial Intelligence) · 4A-FINAL"
    "</div>",
    unsafe_allow_html=True
)
