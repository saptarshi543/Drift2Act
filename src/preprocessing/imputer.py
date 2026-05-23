"""
Drift2Act — Imputation & Feature Scaling Pipeline.

Provides a sequential preprocessing pipeline for clinical time-series
data that is already aggregated to the patient level:

    forward_fill → drop_high_missing → knn_impute → scale_features

Each step is also available as a standalone function.
"""

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from typing import Optional
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Forward-fill within each patient
# ---------------------------------------------------------------------------
def forward_fill_patient(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill missing values within each ``patient_id`` group.

    This is appropriate for *hourly-level* data where the most recent
    observation carries forward until a new reading arrives.

    Parameters
    ----------
    df : pd.DataFrame
        Hourly-level data with a ``patient_id`` column.

    Returns
    -------
    pd.DataFrame
        A copy of *df* with within-patient forward-fill applied.
        The original DataFrame is **not** mutated.
    """
    if "patient_id" not in df.columns:
        logger.warning(
            "No 'patient_id' column found — applying forward-fill globally."
        )
        return df.ffill()

    df = df.copy()
    # Sort to guarantee temporal order within each patient
    if "ICULOS" in df.columns:
        df = df.sort_values(["patient_id", "ICULOS"])

    df = df.groupby("patient_id", group_keys=False).apply(
        lambda g: g.ffill()
    )

    n_remaining = int(df.isna().sum().sum())
    logger.info("After forward-fill: %d NaN values remain.", n_remaining)
    return df


# ---------------------------------------------------------------------------
# 2. Drop columns with excessive missingness
# ---------------------------------------------------------------------------
def drop_high_missing(
    df: pd.DataFrame,
    threshold: float = 0.6,
) -> tuple[pd.DataFrame, list[str]]:
    """Drop features whose fraction of missing values exceeds *threshold*.

    Non-feature columns (``patient_id``, ``phase``, ``SepsisLabel``) are
    never dropped.

    Parameters
    ----------
    df : pd.DataFrame
        Patient-level or hourly-level data.
    threshold : float
        Maximum allowable fraction of NaN values.  Columns exceeding
        this are removed.

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        ``(filtered_df, dropped_feature_names)``
    """
    protect = {"patient_id", "phase", "SepsisLabel", "window_id", "label"}
    missing_frac = df.isna().mean()

    to_drop = [
        col
        for col, frac in missing_frac.items()
        if frac > threshold and col not in protect
    ]

    if to_drop:
        logger.info(
            "Dropping %d columns with >%.0f%% missing: %s",
            len(to_drop),
            threshold * 100,
            to_drop,
        )
    else:
        logger.info("No columns exceed the %.0f%% missing threshold.", threshold * 100)

    df_out = df.drop(columns=to_drop)
    return df_out, to_drop


# ---------------------------------------------------------------------------
# 3. KNN imputation
# ---------------------------------------------------------------------------
def knn_impute(
    df: pd.DataFrame,
    feature_cols: list[str],
    k: int = 5,
) -> pd.DataFrame:
    """Impute remaining NaN values in *feature_cols* using KNN.

    Columns that are **entirely** NaN are filled with **0.0** before KNN
    (scikit-learn's ``KNNImputer`` cannot handle all-NaN columns).

    Parameters
    ----------
    df : pd.DataFrame
        Data to impute.
    feature_cols : list[str]
        Column names to impute.  Other columns are left untouched.
    k : int
        Number of neighbours for ``KNNImputer``.

    Returns
    -------
    pd.DataFrame
        A copy of *df* with NaN values in *feature_cols* imputed.
    """
    df = df.copy()

    # Filter to columns actually present
    present_cols = [c for c in feature_cols if c in df.columns]
    if not present_cols:
        logger.warning("No feature columns found in DataFrame — skipping KNN imputation.")
        return df

    subset = df[present_cols].copy()

    # Handle all-NaN columns: fill with 0 (no neighbours can help)
    all_nan_cols = [c for c in present_cols if subset[c].isna().all()]
    if all_nan_cols:
        logger.warning(
            "All-NaN columns filled with 0.0 before KNN: %s", all_nan_cols
        )
        subset[all_nan_cols] = 0.0

    # If there are still NaN values, run KNN
    if subset.isna().any().any():
        n_samples = len(subset)
        effective_k = min(k, n_samples - 1) if n_samples > 1 else 1
        imputer = KNNImputer(n_neighbors=effective_k, weights="uniform")
        imputed_arr = imputer.fit_transform(subset)
        df[present_cols] = imputed_arr
        logger.info(
            "KNN imputation (k=%d) applied to %d columns.", effective_k, len(present_cols)
        )
    else:
        logger.info("No NaN values remaining — KNN imputation skipped.")
        df[present_cols] = subset.values

    return df


# ---------------------------------------------------------------------------
# 4. Feature scaling
# ---------------------------------------------------------------------------
def scale_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    scaler: Optional[StandardScaler] = None,
) -> tuple[pd.DataFrame, StandardScaler]:
    """Standardise (z-score) *feature_cols* using ``StandardScaler``.

    If a pre-fitted *scaler* is supplied it is used to **transform only**
    (useful for applying the reference scaler to drift phases).  Otherwise
    a new scaler is fitted on the provided data.

    Parameters
    ----------
    df : pd.DataFrame
        Data to scale.
    feature_cols : list[str]
        Columns to standardise.
    scaler : StandardScaler or None
        Pre-fitted scaler.  ``None`` → fit a new one.

    Returns
    -------
    tuple[pd.DataFrame, StandardScaler]
        ``(scaled_df, fitted_scaler)``
    """
    df = df.copy()
    present_cols = [c for c in feature_cols if c in df.columns]

    if not present_cols:
        logger.warning("No feature columns present — scaling skipped.")
        return df, scaler or StandardScaler()

    # Replace any residual infinities with NaN, then fill
    df[present_cols] = df[present_cols].replace([np.inf, -np.inf], np.nan)
    df[present_cols] = df[present_cols].fillna(0.0)

    if scaler is None:
        scaler = StandardScaler()
        df[present_cols] = scaler.fit_transform(df[present_cols])
        logger.info("Fitted StandardScaler on %d features.", len(present_cols))
    else:
        df[present_cols] = scaler.transform(df[present_cols])
        logger.info("Applied existing StandardScaler to %d features.", len(present_cols))

    return df, scaler


# ---------------------------------------------------------------------------
# 5. Full preprocessing pipeline
# ---------------------------------------------------------------------------
def run_preprocessing_pipeline(
    df: pd.DataFrame,
    missing_threshold: float = 0.6,
    knn_k: int = 5,
    scaler: Optional[StandardScaler] = None,
) -> tuple[pd.DataFrame, dict]:
    """Execute the complete preprocessing pipeline.

    Steps
    -----
    1. **Forward-fill** within each patient (hourly data only).
    2. **Drop** columns with > *missing_threshold* fraction NaN.
    3. **KNN-impute** remaining NaN in feature columns.
    4. **Standard-scale** feature columns.

    Parameters
    ----------
    df : pd.DataFrame
        Raw or lightly-processed data (hourly or patient-level).
    missing_threshold : float
        Columns with missingness above this fraction are dropped.
    knn_k : int
        Number of neighbours for KNN imputation.
    scaler : StandardScaler or None
        If provided, the scaler is used for transform-only (no re-fit).

    Returns
    -------
    tuple[pd.DataFrame, dict]
        ``(processed_df, metadata)`` where *metadata* contains:

        - ``"scaler"`` — the fitted ``StandardScaler``
        - ``"dropped_features"`` — list of removed column names
        - ``"feature_cols"`` — list of feature column names that survived
    """
    logger.info("=== Starting preprocessing pipeline ===")
    logger.info("Input shape: %s", df.shape)

    # Step 1: forward-fill (makes sense only for hourly data)
    if "patient_id" in df.columns and "ICULOS" in df.columns:
        df = forward_fill_patient(df)
    else:
        logger.info("Skipping forward-fill (not hourly-level data).")

    # Step 2: drop high-missing columns
    df, dropped = drop_high_missing(df, threshold=missing_threshold)

    # Identify remaining numeric feature columns
    non_feature = {"patient_id", "phase", "SepsisLabel", "window_id", "label"}
    feature_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in non_feature
    ]

    # Step 3: KNN imputation
    df = knn_impute(df, feature_cols=feature_cols, k=knn_k)

    # Step 4: scaling
    df, fitted_scaler = scale_features(df, feature_cols=feature_cols, scaler=scaler)

    metadata = {
        "scaler": fitted_scaler,
        "dropped_features": dropped,
        "feature_cols": feature_cols,
    }

    logger.info("Output shape: %s  |  Features: %d", df.shape, len(feature_cols))
    logger.info("=== Preprocessing pipeline complete ===")
    return df, metadata
