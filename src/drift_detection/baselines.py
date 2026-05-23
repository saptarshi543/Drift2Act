"""
Baseline drift detectors for benchmarking against SADI.

Implements KS-test, PSI, ADWIN (river), Frouros KS, and Evidently
data-drift detectors, plus a unified runner that aggregates results.
"""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def ks_detector(
    ref: pd.DataFrame,
    window: pd.DataFrame,
    feature_cols: list[str],
    p_thresh: float = 0.05,
) -> tuple[bool, list[str], dict]:
    """Run a two-sample Kolmogorov-Smirnov test per feature.

    Compares the empirical distributions of each feature between a
    reference window and a current window.

    Args:
        ref: Reference (baseline) data.
        window: Current window data to test for drift.
        feature_cols: List of feature column names to test.
        p_thresh: Significance level. Features with p-value below this
            threshold are flagged as drifted.

    Returns:
        A tuple of (drift_detected, flagged_features, pvalue_dict) where
        drift_detected is True if *any* feature drifts, flagged_features
        is the list of drifted feature names, and pvalue_dict maps each
        feature name to its KS p-value.
    """
    flagged_features: list[str] = []
    pvalue_dict: dict[str, float] = {}

    for col in feature_cols:
        ref_vals = ref[col].dropna().values
        win_vals = window[col].dropna().values

        # If either sample is empty, skip — cannot run KS test
        if len(ref_vals) == 0 or len(win_vals) == 0:
            logger.warning(
                "Skipping KS test for '%s': insufficient non-null values "
                "(ref=%d, window=%d).",
                col,
                len(ref_vals),
                len(win_vals),
            )
            pvalue_dict[col] = 1.0
            continue

        _, p_value = ks_2samp(ref_vals, win_vals)
        pvalue_dict[col] = float(p_value)

        if p_value < p_thresh:
            flagged_features.append(col)

    drift_detected = len(flagged_features) > 0

    if drift_detected:
        logger.info(
            "KS detector: drift detected in %d / %d features.",
            len(flagged_features),
            len(feature_cols),
        )
    else:
        logger.debug("KS detector: no drift detected.")

    return drift_detected, flagged_features, pvalue_dict


def compute_psi(
    ref_values: np.ndarray,
    cur_values: np.ndarray,
    bins: int = 10,
) -> float:
    """Compute the Population Stability Index between two distributions.

    Uses equal-width histogram binning. Empty bins are smoothed with a
    small epsilon to avoid log(0) and division by zero.

    Args:
        ref_values: 1-D array of reference values.
        cur_values: 1-D array of current values.
        bins: Number of histogram bins.

    Returns:
        The PSI score (non-negative float). Values below 0.1 suggest
        insignificant change, 0.1-0.2 moderate, and above 0.2 significant.
    """
    eps = 1e-6

    ref_values = np.asarray(ref_values, dtype=np.float64)
    cur_values = np.asarray(cur_values, dtype=np.float64)

    # Edge case: empty arrays
    if ref_values.size == 0 or cur_values.size == 0:
        logger.warning("compute_psi received empty array(s); returning 0.0.")
        return 0.0

    # Determine bin edges from the reference distribution
    combined_min = min(ref_values.min(), cur_values.min())
    combined_max = max(ref_values.max(), cur_values.max())

    # Handle constant feature (min == max)
    if combined_min == combined_max:
        return 0.0

    bin_edges = np.linspace(combined_min, combined_max, bins + 1)
    # Ensure the last bin edge captures the max value
    bin_edges[-1] = combined_max + eps

    ref_counts = np.histogram(ref_values, bins=bin_edges)[0].astype(np.float64)
    cur_counts = np.histogram(cur_values, bins=bin_edges)[0].astype(np.float64)

    # Convert to proportions, adding epsilon to avoid zeros
    ref_pct = (ref_counts + eps) / (ref_values.size + eps * bins)
    cur_pct = (cur_counts + eps) / (cur_values.size + eps * bins)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))

    return float(psi)


def psi_detector(
    ref: pd.DataFrame,
    window: pd.DataFrame,
    feature_cols: list[str],
    psi_thresh: float = 0.2,
) -> tuple[bool, list[str], dict]:
    """Compute PSI per feature and flag those exceeding the threshold.

    Args:
        ref: Reference data.
        window: Current window data.
        feature_cols: Feature columns to evaluate.
        psi_thresh: PSI threshold above which a feature is considered drifted.

    Returns:
        A tuple of (drift_detected, flagged_features, psi_scores_dict).
    """
    flagged_features: list[str] = []
    psi_scores_dict: dict[str, float] = {}

    for col in feature_cols:
        ref_vals = ref[col].dropna().values
        win_vals = window[col].dropna().values

        if len(ref_vals) == 0 or len(win_vals) == 0:
            logger.warning(
                "Skipping PSI for '%s': insufficient non-null values.", col
            )
            psi_scores_dict[col] = 0.0
            continue

        psi_val = compute_psi(ref_vals, win_vals)
        psi_scores_dict[col] = psi_val

        if psi_val > psi_thresh:
            flagged_features.append(col)

    drift_detected = len(flagged_features) > 0

    if drift_detected:
        logger.info(
            "PSI detector: drift detected in %d / %d features.",
            len(flagged_features),
            len(feature_cols),
        )
    else:
        logger.debug("PSI detector: no drift detected.")

    return drift_detected, flagged_features, psi_scores_dict


def adwin_on_predictions(
    pred_stream: np.ndarray,
    delta: float = 0.002,
) -> list[bool]:
    """Run ADWIN change detection on a stream of prediction values.

    ADWIN maintains a variable-length window that shrinks when a change
    in the mean is detected with sufficient statistical confidence.

    Args:
        pred_stream: 1-D array of prediction values (probabilities or
            continuous scores) ordered chronologically.
        delta: Confidence parameter for ADWIN. Smaller values make the
            detector less sensitive (fewer false alarms).

    Returns:
        A list of booleans aligned with pred_stream, where True at
        position i indicates a drift was detected at that time step.
        Returns an empty list if river is not installed.
    """
    try:
        from river.drift import ADWIN  # type: ignore[import-untyped]
    except ImportError:
        logger.warning(
            "river is not installed — ADWIN detector unavailable. "
            "Install with: pip install river"
        )
        return []

    pred_stream = np.asarray(pred_stream, dtype=np.float64)
    if pred_stream.size == 0:
        return []

    adwin = ADWIN(delta=delta)
    drift_flags: list[bool] = []

    for value in pred_stream:
        adwin.update(float(value))
        drift_flags.append(adwin.drift_detected)

    n_detections = sum(drift_flags)
    if n_detections > 0:
        logger.info(
            "ADWIN detected drift at %d / %d time steps.",
            n_detections,
            len(pred_stream),
        )
    else:
        logger.debug("ADWIN: no drift detected in prediction stream.")

    return drift_flags


def frouros_ks_detector(
    ref_data: np.ndarray,
    window_data: np.ndarray,
) -> tuple[bool, float]:
    """Run a KS-based drift test using the Frouros library.

    Frouros provides a research-grade implementation of the two-sample
    Kolmogorov-Smirnov test with built-in significance testing.

    Args:
        ref_data: 1-D array of reference values.
        window_data: 1-D array of current values.

    Returns:
        A tuple of (drift_detected, statistic). drift_detected is True
        if the KS statistic exceeds the critical value (p < 0.05).
        Returns (False, 0.0) if frouros is not installed.
    """
    try:
        from frouros.detectors.data_drift import KSTest  # type: ignore[import-untyped]
    except ImportError:
        logger.warning(
            "frouros is not installed — Frouros KS detector unavailable. "
            "Install with: pip install frouros"
        )
        return False, 0.0

    ref_data = np.asarray(ref_data, dtype=np.float64).reshape(-1, 1)
    window_data = np.asarray(window_data, dtype=np.float64).reshape(-1, 1)

    if ref_data.size == 0 or window_data.size == 0:
        logger.warning(
            "frouros_ks_detector received empty array(s); returning no drift."
        )
        return False, 0.0

    try:
        detector = KSTest()
        detector.fit(ref_data)
        result = detector.compare(window_data)

        # Frouros returns (statistic, p_value) or a result object depending
        # on the version. Handle both gracefully.
        if hasattr(result, "__iter__"):
            # result is a tuple-like: (statistic, p_value) or similar
            stat_value = float(result[0])
            p_value = float(result[1]) if len(result) > 1 else 0.0
        else:
            stat_value = float(result)
            p_value = 0.0

        drift_detected = p_value < 0.05 if p_value > 0 else False
        logger.debug(
            "Frouros KS: statistic=%.4f, p_value=%.4f, drift=%s",
            stat_value,
            p_value,
            drift_detected,
        )
        return drift_detected, stat_value

    except Exception as exc:
        logger.error("Frouros KS detector failed: %s", exc)
        return False, 0.0


def evidently_drift_report(
    ref: pd.DataFrame,
    window: pd.DataFrame,
    feature_cols: list[str],
) -> dict:
    """Generate a data drift report using the Evidently library.

    Uses Evidently's DataDriftPreset to produce per-feature drift flags
    based on statistical tests selected automatically per feature type.

    Args:
        ref: Reference data (must contain feature_cols).
        window: Current window data (must contain feature_cols).
        feature_cols: Columns to include in the drift analysis.

    Returns:
        A dict mapping feature names to booleans (True = drift detected).
        Returns an empty dict if evidently is not installed or on error.
    """
    try:
        from evidently.report import Report  # type: ignore[import-untyped]
        from evidently.metric_preset import DataDriftPreset  # type: ignore[import-untyped]
    except ImportError:
        logger.warning(
            "evidently is not installed — Evidently drift report unavailable. "
            "Install with: pip install evidently"
        )
        return {}

    try:
        ref_subset = ref[feature_cols].copy()
        window_subset = window[feature_cols].copy()

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=ref_subset, current_data=window_subset)

        result_json = report.as_dict()

        drift_flags: dict[str, bool] = {}

        # Navigate the Evidently result structure to extract per-feature drift
        metrics = result_json.get("metrics", [])
        for metric in metrics:
            metric_result = metric.get("result", {})
            drift_by_columns = metric_result.get("drift_by_columns", {})
            for col_name, col_info in drift_by_columns.items():
                if col_name in feature_cols:
                    drift_flags[col_name] = col_info.get(
                        "drift_detected", False
                    )

        logger.info(
            "Evidently report: %d / %d features drifted.",
            sum(drift_flags.values()),
            len(feature_cols),
        )
        return drift_flags

    except Exception as exc:
        logger.error("Evidently drift report failed: %s", exc)
        return {}


def run_all_baseline_detectors(
    ref: pd.DataFrame,
    window: pd.DataFrame,
    feature_cols: list[str],
    pred_ref: Optional[np.ndarray] = None,
    pred_cur: Optional[np.ndarray] = None,
) -> dict:
    """Run all baseline drift detectors and return unified results.

    Always runs KS and PSI detectors. Optionally runs ADWIN on the
    current prediction stream if pred_cur is provided.

    Args:
        ref: Reference data.
        window: Current window data.
        feature_cols: Feature columns to test for drift.
        pred_ref: Optional reference predictions (unused by ADWIN,
            reserved for future methods).
        pred_cur: Optional current prediction stream for ADWIN.

    Returns:
        A dict with keys 'ks', 'psi', and 'adwin', each containing a
        sub-dict with 'drift_detected' (bool), 'flagged_features'
        (list[str]), and method-specific details.
    """
    results: dict = {}

    # --- KS detector ---
    ks_drift, ks_flagged, ks_pvalues = ks_detector(ref, window, feature_cols)
    results["ks"] = {
        "drift_detected": ks_drift,
        "flagged_features": ks_flagged,
        "pvalues": ks_pvalues,
    }

    # --- PSI detector ---
    psi_drift, psi_flagged, psi_scores = psi_detector(ref, window, feature_cols)
    results["psi"] = {
        "drift_detected": psi_drift,
        "flagged_features": psi_flagged,
        "scores": psi_scores,
    }

    # --- ADWIN detector (on prediction stream) ---
    if pred_cur is not None and len(pred_cur) > 0:
        adwin_flags = adwin_on_predictions(pred_cur)
        adwin_drift = any(adwin_flags) if adwin_flags else False
        # Count the number of drift points detected
        n_drift_points = sum(adwin_flags) if adwin_flags else 0
        results["adwin"] = {
            "drift_detected": adwin_drift,
            "flagged_features": [],  # ADWIN operates on predictions, not features
            "n_drift_points": n_drift_points,
            "drift_flags": adwin_flags,
        }
    else:
        results["adwin"] = {
            "drift_detected": False,
            "flagged_features": [],
            "n_drift_points": 0,
            "drift_flags": [],
        }

    # --- Summary ---
    any_drift = any(
        results[method]["drift_detected"] for method in results
    )
    results["summary"] = {
        "any_drift_detected": any_drift,
        "methods_reporting_drift": [
            m for m in ("ks", "psi", "adwin") if results[m]["drift_detected"]
        ],
    }

    logger.info(
        "Baseline detectors summary: drift=%s, methods=%s",
        any_drift,
        results["summary"]["methods_reporting_drift"],
    )

    return results
