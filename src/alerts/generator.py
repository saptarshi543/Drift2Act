"""
Clinical Alert Generator
========================
Produces human-readable, clinician-facing alert blocks summarising
drift severity, intervention actions, SHAP-attributed feature shifts,
fairness concerns, and actionable recommendations.  Alerts can be
persisted as JSON for audit trails.
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


def generate_clinical_alert(
    window_idx: int,
    d_total: float,
    intervention: dict,
    top_features: list[tuple],
    fairness: dict,
    baseline_auprc: float,
    estimated_auprc: float,
) -> str:
    """Generate a formatted ASCII clinical alert block.

    Parameters
    ----------
    window_idx : int
        Index of the current temporal window.
    d_total : float
        Composite SADI drift score.
    intervention : dict
        Output of ``drift2act_decision`` — must contain ``level``,
        ``action``, ``message``.
    top_features : list[tuple]
        Each tuple is ``(feature_name, sadi_score, shap_mean_ref,
        shap_mean_cur)`` for the top drifted features.
    fairness : dict
        Output of ``check_fairness_threshold`` — keys ``dpd_exceeded``,
        ``eod_exceeded``, ``alert_message``.
    baseline_auprc : float
        Baseline AUPRC on reference data.
    estimated_auprc : float
        Estimated AUPRC on current window.

    Returns
    -------
    str
        Multi-line formatted alert string.
    """
    severity = severity_label(d_total)
    auprc_drop = baseline_auprc - estimated_auprc
    auprc_drop_pct = (auprc_drop / baseline_auprc * 100) if baseline_auprc > 0 else 0.0
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    width = 72
    border = "=" * width

    lines: list[str] = [
        border,
        _center("DRIFT2ACT CLINICAL ALERT", width),
        border,
        f"  Timestamp       : {timestamp}",
        f"  Window          : {window_idx}",
        f"  Drift Severity  : {severity} (SADI = {d_total:.4f})",
        f"  Intervention    : Level {intervention.get('level', '?')} — {intervention.get('action', 'UNKNOWN')}",
        "",
        "-" * width,
        _center("PERFORMANCE ESTIMATE", width),
        "-" * width,
        f"  Baseline AUPRC  : {baseline_auprc:.4f}",
        f"  Estimated AUPRC : {estimated_auprc:.4f}",
        f"  AUPRC Drop      : {auprc_drop:.4f} ({auprc_drop_pct:.1f}%)",
        "",
    ]

    # ---- Top drifted features ----
    if top_features:
        lines.append("-" * width)
        lines.append(_center("TOP DRIFTED FEATURES", width))
        lines.append("-" * width)
        lines.append(f"  {'Feature':<25} {'SADI':>8}  {'Direction'}")
        lines.append(f"  {'-------':<25} {'----':>8}  {'---------'}")
        for feat_info in top_features:
            name = feat_info[0]
            sadi = feat_info[1] if len(feat_info) > 1 else 0.0
            shap_ref = feat_info[2] if len(feat_info) > 2 else 0.0
            shap_cur = feat_info[3] if len(feat_info) > 3 else 0.0
            direction = feature_direction(shap_ref, shap_cur)
            lines.append(f"  {name:<25} {sadi:>8.4f}  {direction}")
        lines.append("")

    # ---- Fairness alerts ----
    if fairness.get("dpd_exceeded") or fairness.get("eod_exceeded"):
        lines.append("-" * width)
        lines.append(_center("⚠  FAIRNESS ALERT", width))
        lines.append("-" * width)
        alert_msg = fairness.get("alert_message", "")
        # Wrap long alert messages
        for sentence in alert_msg.split(". "):
            sentence = sentence.strip()
            if sentence:
                if not sentence.endswith("."):
                    sentence += "."
                lines.append(f"  {sentence}")
        lines.append("")

    # ---- Recommendation ----
    lines.append("-" * width)
    lines.append(_center("RECOMMENDATION", width))
    lines.append("-" * width)
    feat_names = [f[0] for f in top_features] if top_features else []
    recommendation = format_recommendation(intervention.get("level", 0), feat_names)
    # Wrap recommendation text at ~68 chars
    for rline in _wrap_text(recommendation, width - 4):
        lines.append(f"  {rline}")
    lines.append("")
    lines.append(border)

    alert_text = "\n".join(lines)
    logger.info(
        "Generated clinical alert for window %d (severity=%s, level=%d).",
        window_idx,
        severity,
        intervention.get("level", -1),
    )
    return alert_text


def severity_label(d_total: float) -> str:
    """Map a composite SADI score to a human-readable severity label.

    Parameters
    ----------
    d_total : float
        Composite SADI drift score.

    Returns
    -------
    str
        One of ``'LOW'``, ``'MODERATE'``, ``'HIGH'``, ``'SEVERE'``.
    """
    if d_total < 0.3:
        return "LOW"
    elif d_total < 0.5:
        return "MODERATE"
    elif d_total < 0.8:
        return "HIGH"
    else:
        return "SEVERE"


def feature_direction(shap_mean_ref: float, shap_mean_cur: float) -> str:
    """Describe the direction of a feature's SHAP-value shift.

    Heuristic thresholds:

    * **FLIPPED**: sign change (positive → negative or vice versa).
    * **SHIFTED UP / DOWN**: magnitude change > 20 %.
    * **STABLE**: otherwise.

    Parameters
    ----------
    shap_mean_ref : float
        Mean SHAP value of the feature in the reference window.
    shap_mean_cur : float
        Mean SHAP value of the feature in the current window.

    Returns
    -------
    str
        Arrow symbol and direction description.
    """
    # Sign flip detection
    if shap_mean_ref != 0 and shap_mean_cur != 0:
        if (shap_mean_ref > 0) != (shap_mean_cur > 0):
            return "⇅ FLIPPED"

    # Relative change
    ref_abs = abs(shap_mean_ref)
    cur_abs = abs(shap_mean_cur)
    denominator = max(ref_abs, 1e-10)
    relative_change = (cur_abs - ref_abs) / denominator

    if relative_change > 0.20:
        return "↑ SHIFTED UP"
    elif relative_change < -0.20:
        return "↓ SHIFTED DOWN"
    else:
        return "⟷ STABLE"


def save_alert_log(alerts: list[dict], path: str) -> None:
    """Persist a list of alert dictionaries to a JSON file.

    Parameters
    ----------
    alerts : list[dict]
        Each dict typically contains keys such as ``window_idx``,
        ``severity``, ``intervention_level``, ``alert_text``, etc.
    path : str
        Output JSON file path.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    # Make all values JSON-serialisable
    serialisable = _make_serialisable(alerts)

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(serialisable, fh, indent=2, ensure_ascii=False)

    logger.info("Saved %d alert(s) to %s", len(alerts), path)


def format_recommendation(
    intervention_level: int,
    top_features: list,
) -> str:
    """Generate clinician-readable recommendation text.

    Parameters
    ----------
    intervention_level : int
        Drift2Act intervention level (0–4).
    top_features : list
        List of feature names (strings) most affected by drift.

    Returns
    -------
    str
        Plain-text recommendation paragraph.
    """
    feature_list = ", ".join(top_features[:5]) if top_features else "N/A"

    if intervention_level == 0:
        return (
            "No action required. The model is performing within expected "
            "parameters. Continue routine monitoring."
        )

    elif intervention_level == 1:
        return (
            f"Minor distributional shift detected in: {feature_list}. "
            "No immediate model changes needed. Increase monitoring "
            "frequency and verify that data ingestion pipelines are "
            "functioning normally. Notify the clinical informatics team."
        )

    elif intervention_level == 2:
        return (
            f"Moderate drift detected in: {feature_list}. "
            "Recommend Platt-scaling recalibration of the model using "
            "the most recent labelled data. Review whether changes in "
            "clinical protocols or lab assay methods may explain the "
            "shift. Schedule a model performance review within 48 hours."
        )

    elif intervention_level == 3:
        return (
            f"Significant drift detected in: {feature_list}. "
            "Recommend partial retraining — update the model using "
            "recent data while retaining the current architecture. "
            "Prioritise retraining on the most-shifted features. "
            "Validate retrained model on a held-out set before "
            "deployment. Escalate to the ML operations team."
        )

    else:  # level 4
        return (
            f"Severe drift detected across multiple features including: "
            f"{feature_list}. Full model retraining is required with "
            "updated feature engineering and data pipelines. Consider "
            "whether the target population or clinical workflows have "
            "changed fundamentally. Revert to a validated fallback model "
            "or clinical decision rule until the retrained model is "
            "validated and approved for deployment."
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _center(text: str, width: int) -> str:
    """Centre-align *text* within *width* characters.

    Parameters
    ----------
    text : str
        Text to centre.
    width : int
        Total line width.

    Returns
    -------
    str
        Centre-padded string.
    """
    return text.center(width)


def _wrap_text(text: str, max_width: int) -> list[str]:
    """Word-wrap *text* to *max_width* characters per line.

    Parameters
    ----------
    text : str
        Input text.
    max_width : int
        Maximum characters per output line.

    Returns
    -------
    list[str]
        Wrapped lines.
    """
    words = text.split()
    lines: list[str] = []
    current_line = ""
    for word in words:
        if current_line and (len(current_line) + 1 + len(word)) > max_width:
            lines.append(current_line)
            current_line = word
        else:
            current_line = f"{current_line} {word}".strip()
    if current_line:
        lines.append(current_line)
    return lines if lines else [""]


def _make_serialisable(obj: object) -> object:
    """Recursively convert numpy/pandas types to JSON-safe Python types.

    Parameters
    ----------
    obj : object
        Any nested structure of dicts, lists, numpy scalars, etc.

    Returns
    -------
    object
        JSON-serialisable equivalent.
    """
    import numpy as np

    if isinstance(obj, dict):
        return {k: _make_serialisable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serialisable(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    else:
        return obj
