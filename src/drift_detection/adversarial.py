"""
Adversarial (domain-classifier) drift detection.

Trains a logistic-regression classifier to distinguish reference from
current-window samples.  A high AUROC signals that the two
distributions are separable — i.e. the data has drifted.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Minimum samples required per fold for stratified cross-validation
_MIN_SAMPLES_PER_FOLD = 2


def adversarial_drift_score(
    ref_data: pd.DataFrame,
    window_data: pd.DataFrame,
    n_cv: int = 5,
) -> float:
    """Compute an adversarial drift score via a domain classifier.

    Assigns label 0 to reference rows and label 1 to current-window
    rows, then trains a Logistic Regression model with cross-validated
    AUROC.  An AUROC near 0.5 means the classifier cannot tell the
    distributions apart (no drift); an AUROC near 1.0 means the
    distributions are highly separable (strong drift).

    Args:
        ref_data: Reference-period feature matrix (numeric columns only).
        window_data: Current-window feature matrix with the same columns.
        n_cv: Number of cross-validation folds.

    Returns:
        Mean cross-validated AUROC (float in [0, 1]).  Returns 0.5 if
        either input is empty or too small for meaningful CV.
    """
    if ref_data.empty or window_data.empty:
        logger.warning(
            "adversarial_drift_score received empty data; returning 0.5."
        )
        return 0.5

    # Build combined dataset with domain labels
    X_ref = ref_data.values.astype(np.float64)
    X_win = window_data.values.astype(np.float64)

    y_ref = np.zeros(len(X_ref), dtype=np.int32)
    y_win = np.ones(len(X_win), dtype=np.int32)

    X = np.vstack([X_ref, X_win])
    y = np.concatenate([y_ref, y_win])

    # Handle NaN values — impute with column medians from the combined set
    col_medians = np.nanmedian(X, axis=0)
    for col_idx in range(X.shape[1]):
        mask = np.isnan(X[:, col_idx])
        if mask.any():
            X[mask, col_idx] = col_medians[col_idx]

    # Replace any remaining NaNs (entire column NaN) with 0
    np.nan_to_num(X, nan=0.0, copy=False)

    # Adaptive fold count: reduce if not enough samples per class
    min_class_size = min(len(X_ref), len(X_win))
    effective_cv = min(n_cv, min_class_size // _MIN_SAMPLES_PER_FOLD)

    if effective_cv < 2:
        logger.warning(
            "Too few samples for cross-validation "
            "(ref=%d, window=%d, need ≥ %d per fold). Returning 0.5.",
            len(X_ref),
            len(X_win),
            _MIN_SAMPLES_PER_FOLD * 2,
        )
        return 0.5

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train domain classifier
    clf = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        random_state=42,
        class_weight="balanced",
    )

    try:
        scores = cross_val_score(
            clf, X_scaled, y, cv=effective_cv, scoring="roc_auc"
        )
        mean_auroc = float(np.mean(scores))
    except Exception as exc:
        logger.error(
            "Adversarial drift scoring failed: %s. Returning 0.5.", exc
        )
        return 0.5

    logger.info(
        "Adversarial drift score: AUROC=%.4f (±%.4f, %d folds)",
        mean_auroc,
        float(np.std(scores)),
        effective_cv,
    )

    return mean_auroc


def adversarial_feature_importance(
    ref_data: pd.DataFrame,
    window_data: pd.DataFrame,
    feature_names: list[str],
) -> dict[str, float]:
    """Identify which features drive distribution shift via a domain classifier.

    Trains a Logistic Regression on all combined data and returns the
    absolute model coefficients as a proxy for each feature's importance
    in distinguishing the two domains.

    Args:
        ref_data: Reference-period feature matrix.
        window_data: Current-window feature matrix (same columns).
        feature_names: Ordered list of feature names matching the columns
            of ref_data and window_data.

    Returns:
        A dict mapping each feature name to its absolute LR coefficient.
        Returns a dict of zeros if inputs are empty or fitting fails.
    """
    zero_result = {name: 0.0 for name in feature_names}

    if ref_data.empty or window_data.empty:
        logger.warning(
            "adversarial_feature_importance received empty data; "
            "returning zero importances."
        )
        return zero_result

    # Build combined dataset
    X_ref = ref_data.values.astype(np.float64)
    X_win = window_data.values.astype(np.float64)

    y_ref = np.zeros(len(X_ref), dtype=np.int32)
    y_win = np.ones(len(X_win), dtype=np.int32)

    X = np.vstack([X_ref, X_win])
    y = np.concatenate([y_ref, y_win])

    # Handle NaN values
    col_medians = np.nanmedian(X, axis=0)
    for col_idx in range(X.shape[1]):
        mask = np.isnan(X[:, col_idx])
        if mask.any():
            X[mask, col_idx] = col_medians[col_idx]

    np.nan_to_num(X, nan=0.0, copy=False)

    # Need at least a few samples of each class
    if len(X_ref) < 2 or len(X_win) < 2:
        logger.warning(
            "Too few samples for feature importance (ref=%d, window=%d).",
            len(X_ref),
            len(X_win),
        )
        return zero_result

    # Scale and fit
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        random_state=42,
        class_weight="balanced",
    )

    try:
        clf.fit(X_scaled, y)
    except Exception as exc:
        logger.error(
            "Adversarial feature importance fitting failed: %s", exc
        )
        return zero_result

    # Extract absolute coefficients
    abs_coefs = np.abs(clf.coef_[0])

    if len(abs_coefs) != len(feature_names):
        logger.error(
            "Coefficient count (%d) does not match feature_names (%d).",
            len(abs_coefs),
            len(feature_names),
        )
        return zero_result

    importance = {
        name: float(coef) for name, coef in zip(feature_names, abs_coefs)
    }

    # Log the top-5 most important features
    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
    logger.info(
        "Adversarial top-5 drift features: %s",
        ", ".join(f"{n}={v:.4f}" for n, v in top_features),
    )

    return importance


def compute_drift_belief_state(
    d_total: float,
    adv_score: float,
    sadi_weight: float = 0.6,
    adv_weight: float = 0.4,
) -> float:
    """Combine SADI D_total and adversarial score into a drift belief state.

    Fuses two complementary drift signals:
    - SADI's D_total: a composite statistical divergence measure in [0, ∞)
      that is clipped to [0, 1] for normalisation.
    - Adversarial AUROC: domain-classifier accuracy in [0.5, 1.0],
      rescaled to [0, 1] so that 0.5 (chance) maps to 0 and 1.0 maps to 1.

    The belief state is a weighted average of these two normalised
    scores, producing a single value in [0, 1] that represents the
    system's confidence that meaningful drift has occurred.

    Args:
        d_total: SADI divergence score (non-negative). Values above 1
            are clipped to 1 for normalisation.
        adv_score: Adversarial AUROC (expected in [0.5, 1.0]).
        sadi_weight: Weight for the normalised SADI score.
        adv_weight: Weight for the normalised adversarial score.

    Returns:
        Drift belief state in [0, 1].  0 = no drift, 1 = maximal drift.

    Raises:
        ValueError: If weights are negative or both zero.
    """
    if sadi_weight < 0 or adv_weight < 0:
        raise ValueError(
            f"Weights must be non-negative, got sadi_weight={sadi_weight}, "
            f"adv_weight={adv_weight}."
        )

    weight_sum = sadi_weight + adv_weight
    if weight_sum == 0:
        raise ValueError("At least one weight must be positive.")

    # Normalise SADI D_total: clip to [0, 1]
    d_normalised = float(np.clip(d_total, 0.0, 1.0))

    # Normalise adversarial AUROC from [0.5, 1.0] → [0, 1]
    # Scores below 0.5 are clipped to 0 (worse than chance = no signal)
    adv_normalised = float(np.clip((adv_score - 0.5) / 0.5, 0.0, 1.0))

    # Weighted combination (normalise weights in case they don't sum to 1)
    belief = (
        sadi_weight * d_normalised + adv_weight * adv_normalised
    ) / weight_sum

    logger.debug(
        "Drift belief state: d_total=%.4f (norm=%.4f), "
        "adv_score=%.4f (norm=%.4f), belief=%.4f",
        d_total,
        d_normalised,
        adv_score,
        adv_normalised,
        belief,
    )

    return float(belief)
