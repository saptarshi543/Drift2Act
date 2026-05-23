"""
Drift2Act Three-Layer Controller
================================
Implements the core decision engine for the Drift2Act system.
Maps SADI drift scores and estimated performance risk to a 5-level
intervention hierarchy (NO_ACTION → FULL_RETRAIN), and provides
evaluation utilities for detection metrics, ablation studies,
and bootstrap significance testing.
"""

import numpy as np
import pandas as pd
import logging
import json
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Intervention level constants
# ---------------------------------------------------------------------------
INTERVENTION_LEVELS = {
    0: "NO_ACTION",
    1: "ALERT",
    2: "RECALIBRATE",
    3: "PARTIAL_RETRAIN",
    4: "FULL_RETRAIN",
}

INTERVENTION_MESSAGES = {
    0: "System operating within normal parameters. No action required.",
    1: "Minor drift detected. Clinical team notified for awareness.",
    2: "Moderate drift detected. Model recalibration recommended.",
    3: "Significant drift detected. Partial model retraining on recent data recommended.",
    4: "Severe drift detected. Full model retraining with updated feature pipeline required.",
}


def drift2act_decision(
    sadi_score: float,
    risk_score: float,
    sadi_threshold: float = 0.3,
    risk_threshold_moderate: float = 0.05,
    risk_threshold_high: float = 0.10,
) -> dict:
    """Determine the intervention level using 5-level decision logic.

    The decision hierarchy escalates based on the composite SADI drift
    score and an estimated performance-risk score:

    * **Level 0 – NO_ACTION**: SADI < threshold AND risk < moderate.
    * **Level 1 – ALERT**: SADI < threshold×1.3 OR risk < moderate.
    * **Level 2 – RECALIBRATE**: SADI < threshold×1.7 OR risk < high.
    * **Level 3 – PARTIAL_RETRAIN**: SADI < threshold×2.2.
    * **Level 4 – FULL_RETRAIN**: everything else.

    Parameters
    ----------
    sadi_score : float
        Composite SADI drift score for the current window.
    risk_score : float
        Estimated performance degradation (baseline_auprc − estimated_auprc).
    sadi_threshold : float, optional
        Base SADI threshold for the lowest intervention band (default 0.3).
    risk_threshold_moderate : float, optional
        Risk score above which moderate concern is raised (default 0.05).
    risk_threshold_high : float, optional
        Risk score above which high concern is raised (default 0.10).

    Returns
    -------
    dict
        Keys: ``level`` (int), ``action`` (str), ``message`` (str),
        ``timestamp`` (ISO-8601 str), ``sadi_score``, ``risk_score``.
    """
    # Level 0 — both metrics below their lowest thresholds
    if sadi_score < sadi_threshold and risk_score < risk_threshold_moderate:
        level = 0
    # Level 1 — early-warning zone
    elif sadi_score < sadi_threshold * 1.3 or risk_score < risk_threshold_moderate:
        level = 1
    # Level 2 — recalibration zone
    elif sadi_score < sadi_threshold * 1.7 or risk_score < risk_threshold_high:
        level = 2
    # Level 3 — partial retrain zone
    elif sadi_score < sadi_threshold * 2.2:
        level = 3
    # Level 4 — full retrain
    else:
        level = 4

    decision = {
        "level": level,
        "action": INTERVENTION_LEVELS[level],
        "message": INTERVENTION_MESSAGES[level],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sadi_score": float(sadi_score),
        "risk_score": float(risk_score),
    }

    logger.info(
        "Drift2Act decision: level=%d (%s) | SADI=%.4f | risk=%.4f",
        level,
        decision["action"],
        sadi_score,
        risk_score,
    )
    return decision


def compute_risk_score(
    baseline_auprc: float,
    estimated_auprc: float,
) -> float:
    """Compute the performance-risk score as the drop from baseline AUPRC.

    Parameters
    ----------
    baseline_auprc : float
        AUPRC measured on the reference (training) data.
    estimated_auprc : float
        AUPRC estimated on the current analysis window (e.g. via NannyML CBPE).

    Returns
    -------
    float
        ``max(0, min(1, baseline_auprc - estimated_auprc))``.
    """
    risk = baseline_auprc - estimated_auprc
    risk = float(np.clip(risk, 0.0, 1.0))
    logger.debug(
        "Risk score: %.4f  (baseline=%.4f, estimated=%.4f)",
        risk,
        baseline_auprc,
        estimated_auprc,
    )
    return risk


def get_phase_label(
    window_idx: int,
    total_windows: int,
    phase1_frac: float = 0.40,
    phase2_frac: float = 0.25,
) -> str:
    """Label a temporal window as belonging to phase 1, 2, or 3.

    The evaluation timeline is split into three contiguous phases:

    * **phase1** (stable): first ``phase1_frac`` fraction of windows.
    * **phase2** (drift onset): next ``phase2_frac`` fraction.
    * **phase3** (post-drift): remainder.

    Parameters
    ----------
    window_idx : int
        Zero-based index of the current window.
    total_windows : int
        Total number of windows in the evaluation run.
    phase1_frac : float, optional
        Fraction of windows assigned to phase 1 (default 0.40).
    phase2_frac : float, optional
        Fraction of windows assigned to phase 2 (default 0.25).

    Returns
    -------
    str
        One of ``'phase1'``, ``'phase2'``, ``'phase3'``.
    """
    if total_windows <= 0:
        logger.warning("total_windows=%d is non-positive; defaulting to 'phase1'.", total_windows)
        return "phase1"

    phase1_end = int(total_windows * phase1_frac)
    phase2_end = int(total_windows * (phase1_frac + phase2_frac))

    if window_idx < phase1_end:
        return "phase1"
    elif window_idx < phase2_end:
        return "phase2"
    else:
        return "phase3"


def run_drift2act_evaluation(
    results_log: list[dict],
    output_path: str = "results/drift2act_results.csv",
) -> pd.DataFrame:
    """Convert a list of per-window result dicts to a DataFrame and persist.

    Parameters
    ----------
    results_log : list[dict]
        Each dict typically contains keys such as ``window_idx``,
        ``sadi_score``, ``risk_score``, ``intervention_level``,
        ``true_phase``, etc.
    output_path : str, optional
        File path for the saved CSV (default ``results/drift2act_results.csv``).

    Returns
    -------
    pd.DataFrame
        The consolidated results table.
    """
    if not results_log:
        logger.warning("Empty results_log; returning empty DataFrame.")
        return pd.DataFrame()

    df = pd.DataFrame(results_log)

    # Ensure output directory exists
    import os

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Saved %d evaluation records to %s", len(df), output_path)
    return df


def compute_detection_metrics(
    results_df: pd.DataFrame,
    detector_col: str,
    threshold: Optional[float] = None,
) -> dict:
    """Compute precision, recall, F1, and detection latency for a drift detector.

    Ground-truth labels are derived from the ``true_phase`` column:
    * ``phase1`` → no drift (negative)
    * ``phase2`` / ``phase3`` → drift present (positive)

    If *threshold* is provided the detector column is binarised as
    ``results_df[detector_col] >= threshold``; otherwise it is assumed
    to already be binary (0/1 or bool).

    Detection latency is the number of windows from the start of
    ``phase2`` to the first positive prediction inside phase2/phase3.

    Parameters
    ----------
    results_df : pd.DataFrame
        Must contain columns ``true_phase`` and *detector_col*.
    detector_col : str
        Column name holding detector scores or binary flags.
    threshold : float, optional
        If given, values ≥ threshold are treated as positive detections.

    Returns
    -------
    dict
        ``precision``, ``recall``, ``f1``, ``detection_latency``.
    """
    if results_df.empty or detector_col not in results_df.columns:
        logger.warning(
            "Cannot compute detection metrics: empty df or missing column '%s'.",
            detector_col,
        )
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "detection_latency": None}

    df = results_df.copy()

    # Ground-truth: drift present in phase2 and phase3
    y_true = df["true_phase"].isin(["phase2", "phase3"]).astype(int).values

    # Predictions
    if threshold is not None:
        y_pred = (df[detector_col].astype(float) >= threshold).astype(int).values
    else:
        y_pred = df[detector_col].astype(int).values

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # Detection latency: windows from phase2 start to first alert in phase2+
    detection_latency: Optional[int] = None
    drift_mask = df["true_phase"].isin(["phase2", "phase3"])
    if drift_mask.any():
        phase2_start_idx = drift_mask.idxmax()  # first True index
        drift_region = df.loc[phase2_start_idx:]
        if threshold is not None:
            alert_mask = drift_region[detector_col].astype(float) >= threshold
        else:
            alert_mask = drift_region[detector_col].astype(bool)
        if alert_mask.any():
            first_alert_idx = alert_mask.idxmax()
            detection_latency = int(first_alert_idx - phase2_start_idx)

    metrics = {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "detection_latency": detection_latency,
    }
    logger.info(
        "Detection metrics [%s]: P=%.4f  R=%.4f  F1=%.4f  latency=%s",
        detector_col,
        precision,
        recall,
        f1,
        detection_latency,
    )
    return metrics


def run_ablation_study(
    results_df: pd.DataFrame,
    sadi_columns: dict,
) -> pd.DataFrame:
    """Compare detection metrics across multiple SADI weight configurations.

    Parameters
    ----------
    results_df : pd.DataFrame
        Full evaluation results; must include ``true_phase`` and each
        column referenced in *sadi_columns*.
    sadi_columns : dict
        Mapping of descriptive config name → column name in *results_df*.
        Example: ``{"equal_weights": "sadi_equal", "shap_heavy": "sadi_shap_heavy"}``.

    Returns
    -------
    pd.DataFrame
        One row per configuration with precision, recall, F1, and
        detection_latency.
    """
    if results_df.empty:
        logger.warning("Empty results_df; ablation study returns empty DataFrame.")
        return pd.DataFrame()

    rows: list[dict] = []
    for config_name, col_name in sadi_columns.items():
        if col_name not in results_df.columns:
            logger.warning("Column '%s' not found in results_df; skipping config '%s'.", col_name, config_name)
            continue
        metrics = compute_detection_metrics(results_df, col_name)
        metrics["config"] = config_name
        metrics["column"] = col_name
        rows.append(metrics)

    ablation_df = pd.DataFrame(rows)
    if not ablation_df.empty:
        ablation_df = ablation_df[["config", "column", "precision", "recall", "f1", "detection_latency"]]
    logger.info("Ablation study completed with %d configurations.", len(rows))
    return ablation_df


def bootstrap_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> np.ndarray:
    """Compute bootstrapped F1 scores for statistical significance testing.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary labels (0/1).
    y_pred : np.ndarray
        Predicted binary labels (0/1).
    n_bootstrap : int, optional
        Number of bootstrap resamples (default 1000).
    seed : int, optional
        Random seed for reproducibility (default 42).

    Returns
    -------
    np.ndarray
        Array of shape ``(n_bootstrap,)`` with one F1 score per resample.
    """
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    n = len(y_true)

    if n == 0:
        logger.warning("Empty arrays passed to bootstrap_f1; returning zeros.")
        return np.zeros(n_bootstrap)

    f1_scores = np.zeros(n_bootstrap, dtype=float)

    for i in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        yt = y_true[idx]
        yp = y_pred[idx]

        tp = np.sum((yp == 1) & (yt == 1))
        fp = np.sum((yp == 1) & (yt == 0))
        fn = np.sum((yp == 0) & (yt == 1))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        f1_scores[i] = f1

    logger.info(
        "Bootstrap F1: mean=%.4f, std=%.4f, 95%% CI=[%.4f, %.4f]",
        np.mean(f1_scores),
        np.std(f1_scores),
        np.percentile(f1_scores, 2.5),
        np.percentile(f1_scores, 97.5),
    )
    return f1_scores
