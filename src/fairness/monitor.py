"""
Fairness Monitor
================
Monitors demographic parity and equalized odds across sensitive
subgroups (primarily age-based) in the sepsis prediction pipeline.
Provides per-subgroup precision / recall / F1 / AUPRC breakdowns
and threshold-based fairness alerts.
"""

import numpy as np
import pandas as pd
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Default age-group bin edges and labels
_DEFAULT_AGE_BINS = [0, 40, 65, 200]
_DEFAULT_AGE_LABELS = ["young_0-40", "middle_40-65", "elderly_65+"]


def compute_fairness_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_feature: np.ndarray,
) -> dict:
    """Compute demographic-parity difference and equalized-odds difference.

    Uses ``fairlearn.metrics`` when available; otherwise falls back to a
    manual implementation so the pipeline never hard-fails on an optional
    dependency.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary labels (0/1).
    y_pred : np.ndarray
        Predicted binary labels (0/1).
    sensitive_feature : np.ndarray
        Categorical sensitive attribute (e.g. age group).

    Returns
    -------
    dict
        ``dpd`` (demographic parity difference),
        ``eod`` (equalized odds difference).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    sensitive_feature = np.asarray(sensitive_feature)

    try:
        from fairlearn.metrics import (
            demographic_parity_difference,
            equalized_odds_difference,
        )

        dpd = float(demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_feature))
        eod = float(equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive_feature))

        logger.info("Fairlearn metrics — DPD=%.4f, EOD=%.4f", dpd, eod)
        return {"dpd": round(dpd, 4), "eod": round(eod, 4)}

    except ImportError:
        logger.warning("fairlearn not installed; computing fairness metrics manually.")
        return _manual_fairness_metrics(y_true, y_pred, sensitive_feature)
    except Exception as exc:
        logger.warning("fairlearn failed (%s); computing manually.", exc)
        return _manual_fairness_metrics(y_true, y_pred, sensitive_feature)


def _manual_fairness_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_feature: np.ndarray,
) -> dict:
    """Manual fallback for fairness metrics when fairlearn is unavailable.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary labels.
    y_pred : np.ndarray
        Predicted binary labels.
    sensitive_feature : np.ndarray
        Categorical sensitive attribute.

    Returns
    -------
    dict
        ``dpd`` and ``eod``.
    """
    groups = np.unique(sensitive_feature)

    if len(groups) < 2:
        logger.warning("Fewer than 2 subgroups; fairness metrics undefined.")
        return {"dpd": 0.0, "eod": 0.0}

    # ---- Demographic Parity Difference ----
    # max|P(Y_hat=1|G=g1) - P(Y_hat=1|G=g2)| across all group pairs
    selection_rates: list[float] = []
    for g in groups:
        mask = sensitive_feature == g
        n_g = mask.sum()
        if n_g == 0:
            continue
        selection_rates.append(float(y_pred[mask].sum()) / n_g)

    dpd = max(selection_rates) - min(selection_rates) if selection_rates else 0.0

    # ---- Equalized Odds Difference ----
    # max difference in TPR and FPR across groups
    tprs: list[float] = []
    fprs: list[float] = []
    for g in groups:
        mask = sensitive_feature == g
        pos = (y_true[mask] == 1)
        neg = (y_true[mask] == 0)
        tp = np.sum((y_pred[mask] == 1) & pos)
        fn = np.sum((y_pred[mask] == 0) & pos)
        fp = np.sum((y_pred[mask] == 1) & neg)
        tn = np.sum((y_pred[mask] == 0) & neg)
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        tprs.append(tpr)
        fprs.append(fpr)

    tpr_diff = (max(tprs) - min(tprs)) if tprs else 0.0
    fpr_diff = (max(fprs) - min(fprs)) if fprs else 0.0
    eod = max(tpr_diff, fpr_diff)

    logger.info("Manual fairness metrics — DPD=%.4f, EOD=%.4f", dpd, eod)
    return {"dpd": round(dpd, 4), "eod": round(eod, 4)}


def create_age_groups(
    ages: np.ndarray,
    bins: Optional[list] = None,
) -> np.ndarray:
    """Bin continuous ages into categorical groups.

    Default groups:

    * **young**: 0 – 40
    * **middle**: 40 – 65
    * **elderly**: 65+

    Parameters
    ----------
    ages : np.ndarray
        Array of patient ages.
    bins : list, optional
        Custom bin edges.  If ``None``, the defaults
        ``[0, 40, 65, 200]`` are used.

    Returns
    -------
    np.ndarray
        String array of group labels (same length as *ages*).
    """
    ages = np.asarray(ages, dtype=float)
    bin_edges = bins if bins is not None else _DEFAULT_AGE_BINS
    labels = _DEFAULT_AGE_LABELS if bins is None else [
        f"group_{i}" for i in range(len(bin_edges) - 1)
    ]

    # Use pd.cut for robust binning
    groups = pd.cut(
        ages,
        bins=bin_edges,
        labels=labels,
        include_lowest=True,
        right=True,
    )

    result = groups.astype(str).values
    # Replace 'nan' entries (out-of-range ages) with 'unknown'
    result[result == "nan"] = "unknown"

    unique, counts = np.unique(result, return_counts=True)
    for g, c in zip(unique, counts):
        logger.debug("Age group '%s': %d patients", g, c)

    return result


def compute_subgroup_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    age_groups: np.ndarray,
) -> pd.DataFrame:
    """Per-subgroup classification metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary labels.
    y_pred : np.ndarray
        Predicted binary labels.
    y_proba : np.ndarray
        Predicted probabilities for the positive class.
    age_groups : np.ndarray
        Categorical subgroup labels.

    Returns
    -------
    pd.DataFrame
        One row per subgroup with columns:
        ``group``, ``n``, ``prevalence``, ``precision``, ``recall``,
        ``f1``, ``auprc``.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_proba = np.asarray(y_proba, dtype=float)
    age_groups = np.asarray(age_groups)

    rows: list[dict] = []
    for group in np.unique(age_groups):
        mask = age_groups == group
        yt = y_true[mask]
        yp = y_pred[mask]
        ypr = y_proba[mask]

        n = int(mask.sum())
        prevalence = float(yt.mean()) if n > 0 else 0.0

        tp = int(np.sum((yp == 1) & (yt == 1)))
        fp = int(np.sum((yp == 1) & (yt == 0)))
        fn = int(np.sum((yp == 0) & (yt == 1)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        # AUPRC
        auprc = _safe_auprc(yt, ypr)

        rows.append({
            "group": group,
            "n": n,
            "prevalence": round(prevalence, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "auprc": round(auprc, 4),
        })

    df = pd.DataFrame(rows)
    logger.info("Subgroup metrics computed for %d groups.", len(df))
    return df


def _safe_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute AUPRC, returning 0.0 on failure or degenerate inputs.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary labels.
    y_score : np.ndarray
        Predicted probabilities.

    Returns
    -------
    float
        Average precision score, or 0.0 if it cannot be computed.
    """
    try:
        from sklearn.metrics import average_precision_score

        if len(np.unique(y_true)) < 2:
            return 0.0
        return float(average_precision_score(y_true, y_score))
    except Exception:
        return 0.0


def check_fairness_threshold(
    dpd: float,
    eod: float,
    dpd_threshold: float = 0.10,
    eod_threshold: float = 0.10,
) -> dict:
    """Check whether fairness metrics exceed acceptable thresholds.

    Parameters
    ----------
    dpd : float
        Demographic parity difference.
    eod : float
        Equalized odds difference.
    dpd_threshold : float, optional
        Maximum acceptable DPD (default 0.10).
    eod_threshold : float, optional
        Maximum acceptable EOD (default 0.10).

    Returns
    -------
    dict
        ``dpd_exceeded`` (bool), ``eod_exceeded`` (bool),
        ``alert_message`` (str — empty if nothing exceeded).
    """
    dpd_exceeded = abs(dpd) > dpd_threshold
    eod_exceeded = abs(eod) > eod_threshold

    messages: list[str] = []
    if dpd_exceeded:
        messages.append(
            f"Demographic Parity Difference ({dpd:.4f}) exceeds threshold ({dpd_threshold:.2f})."
        )
    if eod_exceeded:
        messages.append(
            f"Equalized Odds Difference ({eod:.4f}) exceeds threshold ({eod_threshold:.2f})."
        )

    alert_message = " ".join(messages)

    if alert_message:
        logger.warning("Fairness alert: %s", alert_message)
    else:
        logger.info("Fairness within thresholds (DPD=%.4f, EOD=%.4f).", dpd, eod)

    return {
        "dpd_exceeded": dpd_exceeded,
        "eod_exceeded": eod_exceeded,
        "alert_message": alert_message,
    }
