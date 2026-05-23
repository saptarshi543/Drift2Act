"""
SADI — SHAP-Aware Drift Index.

The novel per-feature drift metric at the heart of Drift2Act:

    SADI(f, t) = α · KL(S_{t-1}(f) ‖ S_t(f))
               + β · |rank_t(f) − rank_{t-1}(f)| / N
               + γ · 𝟙[sign(μ_{t-1}(f)) ≠ sign(μ_t(f))]

The overall system drift score further combines:

    D_total = α · D_SHAP  +  β · D_feature  +  γ · D_confidence

where:
    D_SHAP       = mean SADI of top-10 features
    D_feature    = mean PSI across features
    D_confidence = Wasserstein distance of prediction probability distributions

Default weights: α = 0.5, β = 0.3, γ = 0.2
"""

import numpy as np
from scipy.stats import gaussian_kde, wasserstein_distance
import logging

logger = logging.getLogger(__name__)

# ─── Numerical stability constants ──────────────────────────────────
_EPS = 1e-10          # avoid log(0) in KL / PSI
_MIN_SAMPLES = 5      # minimum samples for KDE estimation


def shap_kl_divergence(
    shap_ref: np.ndarray,
    shap_new: np.ndarray,
    n_points: int = 100,
) -> float:
    """Estimate KL divergence between two SHAP value distributions via KDE.

    Fits kernel density estimates to each SHAP array and evaluates
    KL(P_ref ‖ P_new) on a shared grid spanning both distributions.

    Edge cases handled:
      - Constant arrays (zero variance) → returns 0.0
      - Fewer than ``_MIN_SAMPLES`` points → returns 0.0
      - Zero-density regions → epsilon smoothing

    Args:
        shap_ref: 1-D array of SHAP values from the reference window.
        shap_new: 1-D array of SHAP values from the current window.
        n_points: Number of evaluation points on the KDE grid.

    Returns:
        Non-negative KL divergence estimate (nats).
    """
    shap_ref = np.asarray(shap_ref, dtype=np.float64).ravel()
    shap_new = np.asarray(shap_new, dtype=np.float64).ravel()

    # Guard: too few samples
    if len(shap_ref) < _MIN_SAMPLES or len(shap_new) < _MIN_SAMPLES:
        logger.debug("Too few samples for KDE (%d, %d)", len(shap_ref), len(shap_new))
        return 0.0

    # Guard: constant distributions (zero variance → KDE fails)
    ref_std = np.std(shap_ref)
    new_std = np.std(shap_new)
    if ref_std < _EPS and new_std < _EPS:
        # Both constant — check if they are the same constant
        if np.abs(np.mean(shap_ref) - np.mean(shap_new)) < _EPS:
            return 0.0
        # Different constants → maximum distinguishability, return a
        # finite penalty rather than infinity
        return 10.0
    if ref_std < _EPS or new_std < _EPS:
        # One is constant, the other varies — return moderate divergence
        return 5.0

    # Fit KDEs
    try:
        kde_ref = gaussian_kde(shap_ref, bw_method="scott")
        kde_new = gaussian_kde(shap_new, bw_method="scott")
    except np.linalg.LinAlgError:
        logger.warning("KDE fitting failed (singular matrix); returning 0.0")
        return 0.0

    # Shared evaluation grid spanning the union of both ranges
    lo = min(shap_ref.min(), shap_new.min())
    hi = max(shap_ref.max(), shap_new.max())
    margin = 0.1 * (hi - lo + _EPS)
    grid = np.linspace(lo - margin, hi + margin, n_points)

    p = kde_ref(grid)
    q = kde_new(grid)

    # Epsilon smoothing to avoid log(0)
    p = np.maximum(p, _EPS)
    q = np.maximum(q, _EPS)

    # Normalise to valid probability densities on the grid
    p = p / np.sum(p)
    q = q / np.sum(q)

    # KL(P ‖ Q) = Σ p_i · log(p_i / q_i)
    kl = float(np.sum(p * np.log(p / q)))

    # Clamp to non-negative (numerical noise can cause tiny negatives)
    return max(kl, 0.0)


def compute_psi(
    ref: np.ndarray,
    cur: np.ndarray,
    bins: int = 10,
) -> float:
    """Compute the Population Stability Index between two distributions.

    PSI quantifies how much a feature's distribution has shifted:
        PSI = Σ (p_i − q_i) · ln(p_i / q_i)

    Values:
      - < 0.1 : no significant shift
      - 0.1–0.25 : moderate shift
      - > 0.25 : significant shift

    Args:
        ref: 1-D array from the reference window.
        cur: 1-D array from the current window.
        bins: Number of equal-width bins.

    Returns:
        Non-negative PSI value.
    """
    ref = np.asarray(ref, dtype=np.float64).ravel()
    cur = np.asarray(cur, dtype=np.float64).ravel()

    if len(ref) == 0 or len(cur) == 0:
        return 0.0

    # Determine bin edges from the reference distribution
    lo = min(ref.min(), cur.min())
    hi = max(ref.max(), cur.max())
    if hi - lo < _EPS:
        return 0.0  # all values identical

    edges = np.linspace(lo, hi, bins + 1)
    # Ensure the rightmost edge captures the max
    edges[-1] += _EPS

    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)

    # Convert to proportions with epsilon smoothing
    ref_prop = (ref_counts + _EPS) / (len(ref) + bins * _EPS)
    cur_prop = (cur_counts + _EPS) / (len(cur) + bins * _EPS)

    psi = float(np.sum((cur_prop - ref_prop) * np.log(cur_prop / ref_prop)))
    return max(psi, 0.0)


def compute_sadi(
    fp_ref: dict,
    shap_new_values: np.ndarray,
    feature_names: list[str],
    alpha: float = 0.5,
    beta: float = 0.3,
    gamma: float = 0.2,
) -> tuple[dict, dict]:
    """Compute per-feature SADI scores between a reference fingerprint and new SHAP values.

    SADI(f) = α · KL(S_ref(f) ‖ S_new(f))
            + β · |rank_ref(f) − rank_new(f)| / N
            + γ · 𝟙[sign(μ_ref(f)) ≠ sign(μ_new(f))]

    Args:
        fp_ref: Reference SHAP fingerprint (from ``build_shap_fingerprint``).
        shap_new_values: New SHAP value matrix, shape ``(n_samples, n_features)``.
        feature_names: Ordered list of feature names matching columns of
            ``shap_new_values``.
        alpha: Weight for KL divergence component.
        beta: Weight for rank-shift component.
        gamma: Weight for direction-flip component.

    Returns:
        Tuple of (sadi_scores, components):
          - **sadi_scores**: ``{feature_name: float}`` — composite SADI
            score per feature.
          - **components**: ``{feature_name: {kl, rank_shift, direction_flip, sadi}}``
            — individual components for explainability.
    """
    n_features = len(feature_names)
    if n_features == 0:
        return {}, {}

    shap_new_values = np.asarray(shap_new_values, dtype=np.float64)
    if shap_new_values.shape[1] != n_features:
        raise ValueError(
            f"Column mismatch: shap_new_values has {shap_new_values.shape[1]} "
            f"columns but {n_features} feature names were given."
        )

    # ── Build new-window statistics ──────────────────────────────────
    new_mean_abs = np.mean(np.abs(shap_new_values), axis=0)
    new_mean = np.mean(shap_new_values, axis=0)

    # Compute new ranks (1 = most important)
    new_rank_order = np.argsort(-new_mean_abs)
    new_ranks = np.empty_like(new_rank_order)
    new_ranks[new_rank_order] = np.arange(1, n_features + 1)

    sadi_scores: dict[str, float] = {}
    components: dict[str, dict] = {}

    for i, fname in enumerate(feature_names):
        if fname not in fp_ref or fname == "_meta":
            continue

        ref_entry = fp_ref[fname]

        # ── Component 1: KL divergence of SHAP distributions ────────
        ref_dist = ref_entry.get("distribution", None)
        new_dist = shap_new_values[:, i]

        if ref_dist is not None and len(ref_dist) >= _MIN_SAMPLES:
            kl = shap_kl_divergence(ref_dist, new_dist)
        else:
            kl = 0.0

        # ── Component 2: Normalised rank shift ──────────────────────
        ref_rank = ref_entry.get("rank", 1)
        new_rank = int(new_ranks[i])
        rank_shift = abs(new_rank - ref_rank) / n_features

        # ── Component 3: Direction flip indicator ────────────────────
        ref_mean = ref_entry.get("mean", 0.0)
        new_mean_val = float(new_mean[i])
        direction_flip = 1.0 if _sign_flipped(ref_mean, new_mean_val) else 0.0

        # ── Composite SADI ──────────────────────────────────────────
        sadi = alpha * kl + beta * rank_shift + gamma * direction_flip

        sadi_scores[fname] = sadi
        components[fname] = {
            "kl": kl,
            "rank_shift": rank_shift,
            "direction_flip": direction_flip,
            "sadi": sadi,
            "ref_rank": ref_rank,
            "new_rank": new_rank,
            "ref_mean": ref_mean,
            "new_mean": new_mean_val,
        }

    logger.info(
        "SADI computed for %d features — max SADI: %.4f, mean SADI: %.4f",
        len(sadi_scores),
        max(sadi_scores.values()) if sadi_scores else 0.0,
        float(np.mean(list(sadi_scores.values()))) if sadi_scores else 0.0,
    )
    return sadi_scores, components


def _sign_flipped(a: float, b: float) -> bool:
    """Check if two values have different signs.

    Treats zero as non-negative (no flip from positive to zero or zero
    to positive).

    Args:
        a: First value.
        b: Second value.

    Returns:
        True if a and b have strictly opposite signs.
    """
    return (a > 0 and b < 0) or (a < 0 and b > 0)


def compute_overall_drift_score(
    sadi_scores: dict[str, float],
    psi_scores: dict[str, float],
    pred_proba_ref: np.ndarray,
    pred_proba_cur: np.ndarray,
    alpha: float = 0.5,
    beta: float = 0.3,
    gamma: float = 0.2,
) -> float:
    """Compute the overall system drift score D_total.

    D_total = α · D_SHAP  +  β · D_feature  +  γ · D_confidence

    where:
      - D_SHAP = mean SADI of the top-10 features (by SADI score)
      - D_feature = mean PSI across all monitored features
      - D_confidence = Wasserstein distance between reference and current
        prediction probability distributions

    Args:
        sadi_scores: Per-feature SADI scores from ``compute_sadi``.
        psi_scores: Per-feature PSI scores from ``compute_psi``.
        pred_proba_ref: 1-D array of reference-window predicted probabilities.
        pred_proba_cur: 1-D array of current-window predicted probabilities.
        alpha: Weight for D_SHAP.
        beta: Weight for D_feature.
        gamma: Weight for D_confidence.

    Returns:
        Scalar composite drift score (higher = more drift).
    """
    # ── D_SHAP: mean SADI of top-10 features ────────────────────────
    if sadi_scores:
        sorted_sadi = sorted(sadi_scores.values(), reverse=True)
        top_10 = sorted_sadi[: min(10, len(sorted_sadi))]
        d_shap = float(np.mean(top_10))
    else:
        d_shap = 0.0

    # ── D_feature: mean PSI ─────────────────────────────────────────
    if psi_scores:
        d_feature = float(np.mean(list(psi_scores.values())))
    else:
        d_feature = 0.0

    # ── D_confidence: Wasserstein distance of prediction probs ──────
    pred_ref = np.asarray(pred_proba_ref, dtype=np.float64).ravel()
    pred_cur = np.asarray(pred_proba_cur, dtype=np.float64).ravel()

    if len(pred_ref) == 0 or len(pred_cur) == 0:
        d_confidence = 0.0
    else:
        d_confidence = float(wasserstein_distance(pred_ref, pred_cur))

    d_total = alpha * d_shap + beta * d_feature + gamma * d_confidence

    logger.info(
        "D_total = %.4f  (D_SHAP=%.4f, D_feature=%.4f, D_confidence=%.4f)",
        d_total,
        d_shap,
        d_feature,
        d_confidence,
    )
    return d_total


def sadi_ablation_configs() -> list[tuple[float, float, float, str]]:
    """Return the five SADI ablation configurations for systematic evaluation.

    Each tuple is ``(alpha, beta, gamma, description)``:
      1. KL divergence only — pure distributional shift
      2. Rank shift only — importance reordering
      3. Direction flip only — sign change detection
      4. Equal weights — unbiased combination
      5. Drift2Act SADI (ours) — empirically tuned weights

    Returns:
        List of 5 ablation configuration tuples.
    """
    return [
        (1.0, 0.0, 0.0, "KL only"),
        (0.0, 1.0, 0.0, "Rank shift only"),
        (0.0, 0.0, 1.0, "Direction flip only"),
        (0.33, 0.33, 0.34, "Equal weights"),
        (0.5, 0.3, 0.2, "Drift2Act SADI (ours)"),
    ]
