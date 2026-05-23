"""
NannyML Wrapper — Label-Free Performance Estimation
====================================================
Provides CBPE-based (Confidence-Based Performance Estimation) AUPRC and
AUROC estimation without ground-truth labels, backed by a lightweight
fallback that uses Wasserstein distance between prediction distributions
when NannyML is not installed or fails.
"""

import numpy as np
import pandas as pd
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def estimate_performance_no_labels(
    reference_df: pd.DataFrame,
    analysis_df: pd.DataFrame,
    y_pred_col: str = "y_pred_proba",
    chunk_size: int = 250,
) -> dict:
    """Estimate AUPRC and AUROC on unlabelled data using NannyML CBPE.

    If NannyML is unavailable or raises an error, the function falls back
    to :func:`estimate_performance_fallback`.

    Parameters
    ----------
    reference_df : pd.DataFrame
        Reference (training) data with columns ``y_pred_proba`` and ``y_true``.
    analysis_df : pd.DataFrame
        Analysis (production) data with column ``y_pred_proba``.
        Labels (``y_true``) may be absent.
    y_pred_col : str, optional
        Name of the predicted-probability column (default ``'y_pred_proba'``).
    chunk_size : int, optional
        Number of rows per CBPE chunk (default 250).

    Returns
    -------
    dict
        Keys: ``estimated_auprc``, ``estimated_auroc``,
        ``lower_bound``, ``upper_bound``, ``method``.
    """
    try:
        import nannyml as nml

        logger.info("NannyML available — using CBPE for performance estimation.")

        # Ensure column naming conventions expected by NannyML
        ref = reference_df.copy()
        ana = analysis_df.copy()

        # NannyML CBPE needs predicted probabilities and ground-truth in reference
        estimator = nml.CBPE(
            y_pred_proba=y_pred_col,
            y_pred="y_pred",
            y_true="y_true",
            problem_type="classification_binary",
            metrics=["average_precision", "roc_auc"],
            chunk_size=chunk_size,
        )

        # Fit on reference (labelled) data
        estimator.fit(ref)

        # Estimate on analysis (unlabelled) data
        results = estimator.estimate(ana)
        results_df = results.to_df()

        # Extract latest chunk estimates
        auprc_col = [c for c in results_df.columns if "average_precision" in c.lower() and "value" in c.lower()]
        auroc_col = [c for c in results_df.columns if "roc_auc" in c.lower() and "value" in c.lower()]
        lower_col = [c for c in results_df.columns if "average_precision" in c.lower() and "lower" in c.lower()]
        upper_col = [c for c in results_df.columns if "average_precision" in c.lower() and "upper" in c.lower()]

        estimated_auprc = float(results_df[auprc_col[0]].iloc[-1]) if auprc_col else np.nan
        estimated_auroc = float(results_df[auroc_col[0]].iloc[-1]) if auroc_col else np.nan
        lower_bound = float(results_df[lower_col[0]].iloc[-1]) if lower_col else np.nan
        upper_bound = float(results_df[upper_col[0]].iloc[-1]) if upper_col else np.nan

        # Sanity-clamp to [0, 1]
        estimated_auprc = float(np.clip(estimated_auprc, 0.0, 1.0))
        estimated_auroc = float(np.clip(estimated_auroc, 0.0, 1.0))

        output = {
            "estimated_auprc": estimated_auprc,
            "estimated_auroc": estimated_auroc,
            "lower_bound": float(np.clip(lower_bound, 0.0, 1.0)) if not np.isnan(lower_bound) else None,
            "upper_bound": float(np.clip(upper_bound, 0.0, 1.0)) if not np.isnan(upper_bound) else None,
            "method": "nannyml_cbpe",
        }
        logger.info(
            "CBPE estimates — AUPRC=%.4f, AUROC=%.4f",
            estimated_auprc,
            estimated_auroc,
        )
        return output

    except Exception as exc:
        logger.warning(
            "NannyML CBPE failed (%s: %s); falling back to distribution-shift proxy.",
            type(exc).__name__,
            exc,
        )

        # ---- Fallback: Wasserstein-based proxy ----
        pred_ref = reference_df[y_pred_col].dropna().values
        pred_cur = analysis_df[y_pred_col].dropna().values

        # Try to get a baseline AUPRC from reference labels
        baseline_auprc = _compute_baseline_auprc(reference_df, y_pred_col)

        return estimate_performance_fallback(pred_ref, pred_cur, baseline_auprc)


def estimate_performance_fallback(
    pred_proba_ref: np.ndarray,
    pred_proba_cur: np.ndarray,
    baseline_auprc: float,
) -> dict:
    """Estimate performance degradation when NannyML is unavailable.

    Uses the 1-D Wasserstein distance between reference and current
    prediction distributions as a proxy for AUPRC drop.  The mapping is:

    ``estimated_auprc = baseline_auprc × (1 − min(wasserstein_dist, 1))``

    This is a conservative heuristic: a Wasserstein distance of 1.0
    (maximum possible for probabilities in [0, 1]) implies complete
    performance collapse to zero.

    Parameters
    ----------
    pred_proba_ref : np.ndarray
        Predicted probabilities on the reference data.
    pred_proba_cur : np.ndarray
        Predicted probabilities on the current analysis window.
    baseline_auprc : float
        AUPRC measured on the reference data.

    Returns
    -------
    dict
        Keys: ``estimated_auprc``, ``estimated_auroc``,
        ``lower_bound``, ``upper_bound``, ``method``,
        ``wasserstein_distance``.
    """
    from scipy.stats import wasserstein_distance as _wd

    pred_proba_ref = np.asarray(pred_proba_ref, dtype=float)
    pred_proba_cur = np.asarray(pred_proba_cur, dtype=float)

    # Handle degenerate inputs
    if pred_proba_ref.size == 0 or pred_proba_cur.size == 0:
        logger.warning("Empty prediction array(s); returning baseline as estimate.")
        return {
            "estimated_auprc": float(baseline_auprc),
            "estimated_auroc": None,
            "lower_bound": None,
            "upper_bound": None,
            "method": "fallback_wasserstein",
            "wasserstein_distance": 0.0,
        }

    w_dist = float(_wd(pred_proba_ref, pred_proba_cur))

    # Map Wasserstein distance → AUPRC drop (linear, clamped)
    degradation_factor = min(w_dist, 1.0)
    estimated_auprc = float(np.clip(baseline_auprc * (1.0 - degradation_factor), 0.0, 1.0))

    # Crude confidence bounds: ±5 % of the estimate
    margin = 0.05 * estimated_auprc
    lower_bound = float(np.clip(estimated_auprc - margin, 0.0, 1.0))
    upper_bound = float(np.clip(estimated_auprc + margin, 0.0, 1.0))

    output = {
        "estimated_auprc": estimated_auprc,
        "estimated_auroc": None,  # not estimable without NannyML
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "method": "fallback_wasserstein",
        "wasserstein_distance": round(w_dist, 6),
    }
    logger.info(
        "Fallback estimate — Wasserstein=%.6f, estimated AUPRC=%.4f (baseline=%.4f)",
        w_dist,
        estimated_auprc,
        baseline_auprc,
    )
    return output


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_baseline_auprc(
    reference_df: pd.DataFrame,
    y_pred_col: str,
    y_true_col: str = "y_true",
) -> float:
    """Compute baseline AUPRC from labelled reference data.

    Parameters
    ----------
    reference_df : pd.DataFrame
        Reference data with ground-truth labels.
    y_pred_col : str
        Name of the predicted-probability column.
    y_true_col : str, optional
        Name of the ground-truth label column (default ``'y_true'``).

    Returns
    -------
    float
        Baseline AUPRC, or 0.5 if computation fails.
    """
    try:
        from sklearn.metrics import average_precision_score

        y_true = reference_df[y_true_col].values
        y_score = reference_df[y_pred_col].values
        mask = ~(np.isnan(y_true) | np.isnan(y_score))
        if mask.sum() < 10:
            logger.warning("Too few valid samples to compute baseline AUPRC; using 0.5.")
            return 0.5
        return float(average_precision_score(y_true[mask], y_score[mask]))
    except Exception as exc:
        logger.warning("Could not compute baseline AUPRC (%s); defaulting to 0.5.", exc)
        return 0.5
