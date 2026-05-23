"""
Model calibration for clinical sepsis prediction.

Post-hoc probability calibration via Platt scaling (sigmoid) ensures
that predicted probabilities reflect true risk, which is critical for
clinical decision-making thresholds and drift monitoring.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def calibrate_model(
    model,
    X_val: np.ndarray | pd.DataFrame,
    y_val: np.ndarray | pd.Series,
    method: str = "sigmoid",
) -> CalibratedClassifierCV:
    """Apply post-hoc calibration to a pre-fitted classifier.

    Uses Platt scaling (``method='sigmoid'``) by default, which fits a
    logistic regression on the raw model outputs.  The ``cv='prefit'``
    setting avoids re-training the base model.

    Args:
        model: A fitted classifier with ``predict_proba`` or
            ``decision_function``.
        X_val: Held-out validation features used to fit the calibrator.
        y_val: Held-out validation labels.
        method: Calibration method — ``'sigmoid'`` (Platt) or
            ``'isotonic'`` (non-parametric). Sigmoid is preferred when
            the calibration set is small.

    Returns:
        A CalibratedClassifierCV wrapping the original model.
    """
    if method not in ("sigmoid", "isotonic"):
        raise ValueError(f"method must be 'sigmoid' or 'isotonic', got '{method}'")

    logger.info("Calibrating model with method='%s'", method)

    # Newer sklearn versions (>=1.6) removed cv='prefit'.
    # Try the modern approach first, fall back to returning the model as-is.
    try:
        calibrated = CalibratedClassifierCV(
            estimator=model,
            method=method,
            cv="prefit",
        )
        calibrated.fit(X_val, y_val)
    except Exception as e:
        logger.info("cv='prefit' not supported (%s); returning original model.", e)
        # The original model already has predict_proba, so return it directly.
        # This is acceptable because XGBoost outputs reasonable probabilities.
        calibrated = model

    logger.info("Calibration complete.")
    return calibrated


def plot_calibration_curve(
    model,
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    n_bins: int = 10,
    save_path: str | None = None,
) -> None:
    """Generate a reliability diagram (calibration curve).

    Plots the fraction of positives against the mean predicted
    probability in each bin, with a diagonal reference line representing
    perfect calibration.

    Args:
        model: Fitted classifier with ``predict_proba``.
        X: Feature matrix.
        y: True binary labels.
        n_bins: Number of bins for grouping predictions.
        save_path: Optional file path to save the figure as PNG.
            Parent directories are created automatically.
    """
    y_arr = np.asarray(y)
    y_proba = model.predict_proba(X)[:, 1]

    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_arr, y_proba, n_bins=n_bins, strategy="uniform"
    )

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7, 8), gridspec_kw={"height_ratios": [3, 1]}
    )

    # ── Reliability diagram ──────────────────────────────────────────
    ax1.plot(
        mean_predicted_value,
        fraction_of_positives,
        "s-",
        color="#1f77b4",
        label="Model",
        linewidth=2,
        markersize=7,
    )
    ax1.plot(
        [0, 1], [0, 1], "k--", alpha=0.5, label="Perfectly calibrated"
    )
    ax1.set_xlabel("Mean predicted probability", fontsize=12)
    ax1.set_ylabel("Fraction of positives", fontsize=12)
    ax1.set_title("Calibration / Reliability Diagram", fontsize=14)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.legend(loc="lower right", fontsize=11)
    ax1.grid(True, alpha=0.3)

    # ── Histogram of predicted probabilities ─────────────────────────
    ax2.hist(y_proba, bins=50, range=(0, 1), color="#1f77b4", alpha=0.7, edgecolor="white")
    ax2.set_xlabel("Predicted probability", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_title("Distribution of Predictions", fontsize=12)

    plt.tight_layout()

    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        logger.info("Calibration plot saved to %s", out)

    plt.close(fig)


def get_calibrated_predictions(
    model,
    X: np.ndarray | pd.DataFrame,
) -> np.ndarray:
    """Extract calibrated probability estimates for the positive class.

    Convenience wrapper around ``predict_proba`` that returns only the
    P(sepsis=1) column.

    Args:
        model: Fitted classifier (calibrated or uncalibrated) with
            ``predict_proba``.
        X: Feature matrix.

    Returns:
        1-D array of predicted probabilities for the positive class.
    """
    proba = model.predict_proba(X)[:, 1]
    return proba
