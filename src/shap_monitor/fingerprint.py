"""
SHAP-based model fingerprinting for concept drift monitoring.

A SHAP fingerprint captures per-feature importance statistics (mean
absolute SHAP, signed mean, std, rank, raw distribution) at a given
point in time.  Comparing fingerprints across temporal windows powers
the SADI drift metric.
"""

import shap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pickle
import logging
import copy
from pathlib import Path

logger = logging.getLogger(__name__)


def _extract_tree_model(model):
    """Unwrap a CalibratedClassifierCV or pipeline to get the tree-based estimator.

    SHAP's TreeExplainer needs the raw tree model, not a calibration
    wrapper.  This function handles:
      - Raw XGBoost / LightGBM / sklearn tree estimators (returned as-is)
      - CalibratedClassifierCV (extracts the base estimator from the
        first calibrated classifier)

    Args:
        model: A fitted model, potentially wrapped.

    Returns:
        The underlying tree-based estimator.

    Raises:
        TypeError: If the base estimator cannot be identified.
    """
    # sklearn.calibration.CalibratedClassifierCV
    if hasattr(model, "calibrated_classifiers_"):
        # CalibratedClassifierCV stores a list of _CalibratedClassifier
        # objects; each has an .estimator attribute
        inner = model.calibrated_classifiers_[0].estimator
        logger.info(
            "Extracted base estimator (%s) from CalibratedClassifierCV",
            type(inner).__name__,
        )
        return inner

    # If the model itself is tree-based, return directly
    if hasattr(model, "get_booster") or hasattr(model, "tree_"):
        return model

    # sklearn Pipeline
    if hasattr(model, "named_steps"):
        # Try the last step
        last = list(model.named_steps.values())[-1]
        return _extract_tree_model(last)

    # Fallback — just return and let TreeExplainer raise if unsupported
    logger.warning(
        "Could not identify a tree estimator inside %s; passing as-is",
        type(model).__name__,
    )
    return model


def compute_shap_values(
    model,
    X: pd.DataFrame,
    feature_names: list[str],
) -> np.ndarray:
    """Compute SHAP values using TreeExplainer with interventional perturbation.

    Args:
        model: Fitted tree-based model (or CalibratedClassifierCV wrapping one).
        X: Feature DataFrame — rows are observations, columns are features.
        feature_names: Ordered list of feature names matching ``X.columns``.

    Returns:
        2-D numpy array of shape ``(n_samples, n_features)`` with SHAP
        values for the positive class.
    """
    tree_model = _extract_tree_model(model)

    logger.info(
        "Computing SHAP values for %d samples × %d features via TreeExplainer",
        X.shape[0],
        len(feature_names),
    )

    # Ensure column order matches feature_names
    X_ordered = X[feature_names] if isinstance(X, pd.DataFrame) else X

    explainer = shap.TreeExplainer(
        tree_model,
        data=X_ordered,
        feature_perturbation="interventional",
    )
    shap_vals = explainer.shap_values(X_ordered)

    # For binary classifiers, shap_values may return a list [neg, pos]
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]  # positive class

    shap_vals = np.asarray(shap_vals, dtype=np.float64)
    logger.info("SHAP values shape: %s", shap_vals.shape)
    return shap_vals


def build_shap_fingerprint(
    shap_values: np.ndarray,
    feature_names: list[str],
) -> dict:
    """Build a statistical fingerprint from a SHAP value matrix.

    For each feature the fingerprint stores:
      - ``mean_abs``: mean of |SHAP| — global importance
      - ``mean``: signed mean SHAP — average direction of effect
      - ``std``: standard deviation — spread of contributions
      - ``distribution``: raw 1-D SHAP array (for KDE-based comparisons)
      - ``rank``: importance rank (1 = most important, descending by mean_abs)

    Args:
        shap_values: Array of shape ``(n_samples, n_features)``.
        feature_names: Feature names corresponding to columns.

    Returns:
        Dictionary keyed by feature name, each containing the statistics
        above.  Also includes a ``_meta`` key with global info.
    """
    if shap_values.shape[1] != len(feature_names):
        raise ValueError(
            f"Column count mismatch: shap_values has {shap_values.shape[1]} "
            f"columns but {len(feature_names)} feature names were given."
        )

    # Compute per-feature statistics
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    mean_signed = np.mean(shap_values, axis=0)
    std_vals = np.std(shap_values, axis=0)

    # Rank features by mean_abs descending (rank 1 = most important)
    rank_order = np.argsort(-mean_abs)  # indices sorted by descending mean_abs
    ranks = np.empty_like(rank_order)
    ranks[rank_order] = np.arange(1, len(rank_order) + 1)

    fingerprint: dict = {}
    for i, name in enumerate(feature_names):
        fingerprint[name] = {
            "mean_abs": float(mean_abs[i]),
            "mean": float(mean_signed[i]),
            "std": float(std_vals[i]),
            "distribution": shap_values[:, i].copy(),
            "rank": int(ranks[i]),
        }

    fingerprint["_meta"] = {
        "n_samples": shap_values.shape[0],
        "n_features": len(feature_names),
        "feature_names": list(feature_names),
    }

    logger.info(
        "Built SHAP fingerprint — %d features, %d samples",
        len(feature_names),
        shap_values.shape[0],
    )
    return fingerprint


def save_fingerprint(fingerprint: dict, path: str) -> None:
    """Save a SHAP fingerprint to disk, with distributions stored separately.

    The main pickle contains the summary statistics (mean_abs, mean, std,
    rank) without the raw distribution arrays, keeping the file compact.
    A companion ``*_distributions.pkl`` file stores the raw arrays for
    KDE-based drift comparisons.

    Args:
        fingerprint: Fingerprint dict as returned by ``build_shap_fingerprint``.
        path: Destination file path for the summary pickle.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Separate distributions from summary statistics
    fp_summary = {}
    distributions = {}

    for key, val in fingerprint.items():
        if key == "_meta":
            fp_summary[key] = val
            continue
        # Deep copy to avoid mutating the caller's dict
        entry = dict(val)
        distributions[key] = entry.pop("distribution", None)
        fp_summary[key] = entry

    with open(out, "wb") as f:
        pickle.dump(fp_summary, f, protocol=pickle.HIGHEST_PROTOCOL)

    dist_path = out.with_name(out.stem + "_distributions" + out.suffix)
    with open(dist_path, "wb") as f:
        pickle.dump(distributions, f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info("Fingerprint saved — summary: %s, distributions: %s", out, dist_path)


def load_fingerprint(path: str) -> dict:
    """Load a SHAP fingerprint from disk, merging distributions if present.

    Automatically looks for the companion ``*_distributions.pkl`` file
    and merges the raw arrays back into the fingerprint dict.

    Args:
        path: Path to the summary pickle file.

    Returns:
        Full fingerprint dict with distributions (if the companion file
        exists).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Fingerprint file not found: {p}")

    with open(p, "rb") as f:
        fp = pickle.load(f)

    # Try to load companion distributions
    dist_path = p.with_name(p.stem + "_distributions" + p.suffix)
    if dist_path.exists():
        with open(dist_path, "rb") as f:
            distributions = pickle.load(f)
        for key in fp:
            if key == "_meta":
                continue
            if key in distributions and distributions[key] is not None:
                fp[key]["distribution"] = distributions[key]
        logger.info("Loaded fingerprint with distributions from %s", p)
    else:
        logger.info("Loaded fingerprint (no distributions file) from %s", p)

    return fp


def plot_shap_beeswarm(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    feature_names: list[str],
    save_path: str | None = None,
    dpi: int = 150,
) -> None:
    """Generate a SHAP beeswarm plot showing feature contributions.

    Each point is one observation; colour encodes the original feature
    value (red = high, blue = low), and position on the x-axis shows
    the SHAP value (impact on prediction).

    Args:
        shap_values: Array of shape ``(n_samples, n_features)``.
        X: Original feature values (for colouring).
        feature_names: Feature names.
        save_path: Optional path to save the figure.
        dpi: Dots-per-inch for the saved figure.
    """
    # Ensure column order
    X_plot = X[feature_names] if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=feature_names)

    # Create a shap.Explanation object for the beeswarm API
    explanation = shap.Explanation(
        values=shap_values,
        data=X_plot.values,
        feature_names=feature_names,
    )

    fig = plt.figure(figsize=(10, max(6, len(feature_names) * 0.35)))
    shap.plots.beeswarm(explanation, show=False, max_display=len(feature_names))

    plt.title("SHAP Beeswarm — Feature Contributions", fontsize=14)
    plt.tight_layout()

    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        logger.info("Beeswarm plot saved to %s", out)

    plt.close(fig)


def plot_shap_bar_comparison(
    fp_ref: dict,
    fp_current: dict,
    feature_names: list[str],
    top_n: int = 15,
    save_path: str | None = None,
    dpi: int = 150,
) -> None:
    """Create a grouped bar chart comparing reference vs. current SHAP importance.

    Displays the top-N features by reference importance side-by-side with
    their current-window importance, making drift visually obvious.

    Args:
        fp_ref: Reference (Phase 1) SHAP fingerprint dict.
        fp_current: Current window SHAP fingerprint dict.
        feature_names: Full list of feature names.
        top_n: Number of top features to display.
        save_path: Optional path to save the figure.
        dpi: Dots-per-inch for the saved figure.
    """
    # Collect mean_abs for both fingerprints
    ref_importance = {
        f: fp_ref[f]["mean_abs"] for f in feature_names if f in fp_ref
    }
    cur_importance = {
        f: fp_current[f]["mean_abs"] for f in feature_names if f in fp_current
    }

    # Sort by reference importance descending, take top_n
    sorted_features = sorted(ref_importance, key=ref_importance.get, reverse=True)[:top_n]
    sorted_features = sorted_features[::-1]  # Reverse for horizontal bar ordering

    ref_vals = [ref_importance.get(f, 0.0) for f in sorted_features]
    cur_vals = [cur_importance.get(f, 0.0) for f in sorted_features]

    y_pos = np.arange(len(sorted_features))
    bar_height = 0.35

    fig, ax = plt.subplots(figsize=(10, max(5, len(sorted_features) * 0.45)))

    ax.barh(
        y_pos - bar_height / 2,
        ref_vals,
        bar_height,
        label="Phase 1 (Reference)",
        color="#1f77b4",
        alpha=0.85,
        edgecolor="white",
    )
    ax.barh(
        y_pos + bar_height / 2,
        cur_vals,
        bar_height,
        label="Current Window",
        color="#ff7f0e",
        alpha=0.85,
        edgecolor="white",
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_features, fontsize=11)
    ax.set_xlabel("Mean |SHAP value|", fontsize=12)
    ax.set_title("SHAP Importance — Reference vs. Current Window", fontsize=14)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        logger.info("Bar comparison plot saved to %s", out)

    plt.close(fig)
