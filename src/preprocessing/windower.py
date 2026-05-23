"""
Drift2Act — Temporal Windowing & Concept Drift Injection.

Provides utilities to:

* Build sliding observation/prediction windows from hourly data.
* Iterate over patient-level data in configurable streaming windows.
* Inject **gradual** (phase 2) and **severe** (phase 3) concept drift.
* Persist drift ground-truth metadata as JSON.
"""

import json
import math
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Iterator
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Sliding observation / prediction windows
# ---------------------------------------------------------------------------
def build_temporal_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    obs_hours: int = 6,
    pred_hours: int = 4,
) -> pd.DataFrame:
    """Build per-patient sliding windows for early-warning prediction.

    For each patient the function slides an *obs_hours*-wide observation
    window one hour at a time and computes **mean, std, min, max** of every
    feature in *feature_cols*.  The binary label for each window is **1**
    if sepsis onset occurs within the next *pred_hours* after the window,
    and **0** otherwise.

    Parameters
    ----------
    df : pd.DataFrame
        Hourly-level data, must contain ``patient_id``, ``ICULOS``, and
        ``SepsisLabel``.
    feature_cols : list[str]
        Columns to aggregate inside each window.
    obs_hours : int
        Width of the observation window in hours.
    pred_hours : int
        How many hours ahead to look for sepsis onset (label horizon).

    Returns
    -------
    pd.DataFrame
        One row per window with columns:
        ``patient_id``, ``window_id``, feature summary statistics
        (``<feat>_mean``, ``<feat>_std``, ``<feat>_min``, ``<feat>_max``),
        and ``label``.
    """
    if "patient_id" not in df.columns:
        raise ValueError("DataFrame must contain a 'patient_id' column.")

    # Ensure available feature columns
    available = [c for c in feature_cols if c in df.columns]
    if not available:
        raise ValueError("None of the requested feature_cols are present.")

    records: list[dict] = []
    window_counter = 0

    grouped = df.sort_values(["patient_id", "ICULOS"]).groupby("patient_id")

    for pid, patient_df in grouped:
        patient_df = patient_df.reset_index(drop=True)
        n_rows = len(patient_df)

        if n_rows < obs_hours:
            # Patient too short — make a single window from all available data
            obs_block = patient_df[available]
            row: dict = {"patient_id": pid, "window_id": window_counter}
            for feat in available:
                vals = obs_block[feat].dropna()
                row[f"{feat}_mean"] = vals.mean() if len(vals) else np.nan
                row[f"{feat}_std"] = vals.std() if len(vals) > 1 else 0.0
                row[f"{feat}_min"] = vals.min() if len(vals) else np.nan
                row[f"{feat}_max"] = vals.max() if len(vals) else np.nan

            # Label: any sepsis in remaining hours
            if "SepsisLabel" in patient_df.columns:
                future = patient_df["SepsisLabel"].values
                row["label"] = int(future.max()) if len(future) else 0
            else:
                row["label"] = 0

            records.append(row)
            window_counter += 1
            continue

        for start in range(0, n_rows - obs_hours + 1):
            end = start + obs_hours  # exclusive
            obs_block = patient_df.iloc[start:end][available]

            row = {"patient_id": pid, "window_id": window_counter}
            for feat in available:
                vals = obs_block[feat].dropna()
                row[f"{feat}_mean"] = vals.mean() if len(vals) else np.nan
                row[f"{feat}_std"] = vals.std() if len(vals) > 1 else 0.0
                row[f"{feat}_min"] = vals.min() if len(vals) else np.nan
                row[f"{feat}_max"] = vals.max() if len(vals) else np.nan

            # Prediction label: sepsis in [end, end + pred_hours)
            if "SepsisLabel" in patient_df.columns:
                pred_end = min(end + pred_hours, n_rows)
                future_labels = patient_df["SepsisLabel"].iloc[end:pred_end].values
                row["label"] = int(future_labels.max()) if len(future_labels) else 0
            else:
                row["label"] = 0

            records.append(row)
            window_counter += 1

    window_df = pd.DataFrame(records)
    logger.info(
        "Built %d windows from %d patients (%d obs_hours, %d pred_hours).",
        len(window_df),
        df["patient_id"].nunique(),
        obs_hours,
        pred_hours,
    )
    return window_df


# ---------------------------------------------------------------------------
# 2. Streaming window iterator
# ---------------------------------------------------------------------------
class StreamingWindowIterator:
    """Iterate over a patient-level DataFrame in configurable windows.

    Yields ``(window_index, window_dataframe)`` pairs.  Useful for
    simulating streaming ingestion in drift-detection experiments.

    Parameters
    ----------
    data : pd.DataFrame
        Patient-level (one row per patient) DataFrame.
    window_size : int
        Number of rows per window.
    step_size : int
        Number of rows to advance between consecutive windows
        (allows overlapping windows when ``step_size < window_size``).
    """

    def __init__(
        self,
        data: pd.DataFrame,
        window_size: int = 200,
        step_size: int = 50,
    ) -> None:
        """Initialise the streaming iterator."""
        if window_size < 1:
            raise ValueError("window_size must be ≥ 1.")
        if step_size < 1:
            raise ValueError("step_size must be ≥ 1.")

        self.data = data.reset_index(drop=True)
        self.window_size = window_size
        self.step_size = step_size
        self._n_windows = max(
            1,
            math.ceil((len(self.data) - self.window_size) / self.step_size) + 1,
        )

    def __len__(self) -> int:
        """Return the total number of windows."""
        return self._n_windows

    def __iter__(self) -> Iterator[tuple[int, pd.DataFrame]]:
        """Yield ``(window_index, window_dataframe)`` pairs."""
        n = len(self.data)
        idx = 0
        window_idx = 0
        while idx < n:
            end = min(idx + self.window_size, n)
            yield window_idx, self.data.iloc[idx:end].copy()
            window_idx += 1
            idx += self.step_size
            # Stop if the next window would start past the data
            if idx >= n:
                break

    def __repr__(self) -> str:
        """Return a human-readable representation."""
        return (
            f"StreamingWindowIterator(n_rows={len(self.data)}, "
            f"window_size={self.window_size}, step_size={self.step_size}, "
            f"n_windows={self._n_windows})"
        )


# ---------------------------------------------------------------------------
# 3. Phase-2 drift injection (gradual respiratory)
# ---------------------------------------------------------------------------
def inject_phase2_drift(
    df: pd.DataFrame,
    feature_cols: list[str],
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """Inject **gradual respiratory drift** (phase 2).

    Affected features (if present): ``O2Sat``, ``Resp``, ``FiO2``,
    ``PaCO2``.  Each is multiplied by a random factor in [0.80, 0.95]
    plus small Gaussian noise.

    Parameters
    ----------
    df : pd.DataFrame
        Patient-level or windowed data to perturb.
    feature_cols : list[str]
        All feature column names (used to locate drift targets).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        ``(drifted_df, drift_log)`` where *drift_log* records the per-
        feature scaling factor and noise magnitude applied.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()

    respiratory_targets = ["O2Sat", "Resp", "FiO2", "PaCO2"]

    drift_log: dict = {
        "phase": "phase2",
        "drift_type": "gradual_respiratory",
        "affected_features": {},
    }

    for base_name in respiratory_targets:
        # Locate matching columns (handles both raw and aggregated names)
        matching = [c for c in feature_cols if c == base_name or c.startswith(f"{base_name}_")]
        matching = [c for c in matching if c in df.columns]

        if not matching:
            continue

        scale_factor = float(rng.uniform(0.80, 0.95))
        noise_std = 0.02

        for col in matching:
            noise = rng.normal(0, noise_std, size=len(df))
            mask = df[col].notna()
            df.loc[mask, col] = df.loc[mask, col] * scale_factor + noise[mask]

        drift_log["affected_features"][base_name] = {
            "columns": matching,
            "scale_factor": round(scale_factor, 4),
            "noise_std": noise_std,
        }

    n_affected = sum(
        len(v["columns"]) for v in drift_log["affected_features"].values()
    )
    logger.info(
        "Phase-2 drift injected: %d base features → %d columns perturbed.",
        len(drift_log["affected_features"]),
        n_affected,
    )
    return df, drift_log


# ---------------------------------------------------------------------------
# 4. Phase-3 drift injection (severe multi-system)
# ---------------------------------------------------------------------------
def inject_phase3_drift(
    df: pd.DataFrame,
    feature_cols: list[str],
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """Inject **severe multi-system drift** (phase 3).

    Drift components
    ~~~~~~~~~~~~~~~~
    1. **Severe respiratory**: ``O2Sat``, ``Resp``, ``FiO2``, ``PaCO2``
       scaled by [0.65, 0.80].
    2. **Inflammatory upward**: ``WBC``, ``Lactate``, ``Fibrinogen``,
       ``Creatinine`` scaled by [1.15, 1.40].
    3. **Correlation breaking**: additional Gaussian noise on ``O2Sat``.
    4. **Label flip**: ~7 % of labels are randomly inverted.

    Parameters
    ----------
    df : pd.DataFrame
        Patient-level or windowed data to perturb.
    feature_cols : list[str]
        All feature column names.
    seed : int
        Random seed.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        ``(drifted_df, drift_log)``
    """
    rng = np.random.default_rng(seed)
    df = df.copy()

    drift_log: dict = {
        "phase": "phase3",
        "drift_type": "severe_multisystem",
        "affected_features": {},
        "label_flip_rate": 0.0,
    }

    # --- 1. Severe respiratory drift ---
    resp_targets = ["O2Sat", "Resp", "FiO2", "PaCO2"]
    for base_name in resp_targets:
        matching = [c for c in feature_cols if c == base_name or c.startswith(f"{base_name}_")]
        matching = [c for c in matching if c in df.columns]
        if not matching:
            continue

        scale_factor = float(rng.uniform(0.65, 0.80))
        noise_std = 0.04

        for col in matching:
            noise = rng.normal(0, noise_std, size=len(df))
            mask = df[col].notna()
            df.loc[mask, col] = df.loc[mask, col] * scale_factor + noise[mask]

        drift_log["affected_features"][base_name] = {
            "columns": matching,
            "scale_factor": round(scale_factor, 4),
            "noise_std": noise_std,
            "component": "severe_respiratory",
        }

    # --- 2. Inflammatory upward drift ---
    inflammatory_targets = ["WBC", "Lactate", "Fibrinogen", "Creatinine"]
    for base_name in inflammatory_targets:
        matching = [c for c in feature_cols if c == base_name or c.startswith(f"{base_name}_")]
        matching = [c for c in matching if c in df.columns]
        if not matching:
            continue

        scale_factor = float(rng.uniform(1.15, 1.40))
        noise_std = 0.03

        for col in matching:
            noise = rng.normal(0, noise_std, size=len(df))
            mask = df[col].notna()
            df.loc[mask, col] = df.loc[mask, col] * scale_factor + noise[mask]

        drift_log["affected_features"][base_name] = {
            "columns": matching,
            "scale_factor": round(scale_factor, 4),
            "noise_std": noise_std,
            "component": "inflammatory_upward",
        }

    # --- 3. Correlation-breaking noise on O2Sat ---
    o2sat_cols = [c for c in feature_cols if c == "O2Sat" or c.startswith("O2Sat_")]
    o2sat_cols = [c for c in o2sat_cols if c in df.columns]
    corr_noise_std = 0.15
    for col in o2sat_cols:
        noise = rng.normal(0, corr_noise_std, size=len(df))
        mask = df[col].notna()
        df.loc[mask, col] = df.loc[mask, col] + noise[mask]

    if o2sat_cols:
        drift_log["affected_features"]["O2Sat_correlation_break"] = {
            "columns": o2sat_cols,
            "additional_noise_std": corr_noise_std,
            "component": "correlation_breaking",
        }

    # --- 4. Label flip (~7 %) ---
    label_col = None
    for candidate in ("SepsisLabel", "label"):
        if candidate in df.columns:
            label_col = candidate
            break

    if label_col is not None:
        flip_rate = 0.07
        flip_mask = rng.random(len(df)) < flip_rate
        n_flipped = int(flip_mask.sum())
        df.loc[flip_mask, label_col] = 1 - df.loc[flip_mask, label_col].astype(int)
        drift_log["label_flip_rate"] = round(n_flipped / max(len(df), 1), 4)
        logger.info("Label flip: %d / %d rows (%.1f%%).", n_flipped, len(df), 100 * n_flipped / len(df))
    else:
        logger.warning("No label column found — skipping label flip.")

    n_affected = sum(
        len(v["columns"])
        for v in drift_log["affected_features"].values()
        if "columns" in v
    )
    logger.info(
        "Phase-3 drift injected: %d components → %d columns perturbed.",
        len(drift_log["affected_features"]),
        n_affected,
    )
    return df, drift_log


# ---------------------------------------------------------------------------
# 5. Save drift ground-truth
# ---------------------------------------------------------------------------
def save_drift_ground_truth(drift_log: dict, path: str) -> None:
    """Persist a drift log dictionary as a JSON file.

    Creates parent directories if they do not exist.

    Parameters
    ----------
    drift_log : dict
        Drift metadata (as returned by ``inject_phase2_drift`` or
        ``inject_phase3_drift``).
    path : str
        Destination file path (e.g. ``"results/drift_ground_truth.json"``).
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(drift_log, f, indent=2, default=str)

    logger.info("Drift ground-truth saved to %s", out_path)
