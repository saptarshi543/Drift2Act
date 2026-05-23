"""
Drift2Act — PhysioNet 2019 Sepsis Challenge Data Loader.

Loads individual .psv patient files, concatenates into a unified DataFrame,
aggregates to patient-level features, assigns temporal phases for drift
simulation, and provides a synthetic data fallback for testing.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature group constants
# ---------------------------------------------------------------------------
VITAL_FEATURES: list[str] = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
]

LAB_FEATURES: list[str] = [
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Glucose",
    "Lactate", "Magnesium", "Phosphate", "Potassium", "Hct", "Hgb",
    "PTT", "WBC", "Fibrinogen", "Platelets", "TroponinI", "Bilirubin_total",
]

DEMO_FEATURES: list[str] = [
    "Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS",
]

ALL_FEATURES: list[str] = VITAL_FEATURES + LAB_FEATURES + DEMO_FEATURES
TARGET: str = "SepsisLabel"


# ---------------------------------------------------------------------------
# 1. Load a single .psv file
# ---------------------------------------------------------------------------
def load_single_psv(filepath: Path) -> pd.DataFrame:
    """Read one pipe-separated (.psv) patient file and attach *patient_id*.

    Parameters
    ----------
    filepath : Path
        Absolute or relative path to a ``*.psv`` file.  The patient
        identifier is extracted from the stem of the filename
        (e.g. ``p000123.psv`` → ``"p000123"``).

    Returns
    -------
    pd.DataFrame
        The patient's hourly readings with an additional ``patient_id``
        column.

    Raises
    ------
    FileNotFoundError
        If *filepath* does not exist.
    pd.errors.EmptyDataError
        If the file is empty or unparseable.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"PSV file not found: {filepath}")

    df = pd.read_csv(filepath, sep="|")
    df["patient_id"] = filepath.stem
    return df


# ---------------------------------------------------------------------------
# 2. Batch-load all .psv files from a directory
# ---------------------------------------------------------------------------
def load_physionet(
    data_dir: str = "data/raw/training",
    max_patients: Optional[int] = None,
) -> pd.DataFrame:
    """Load all ``.psv`` files from *data_dir* into a single DataFrame.

    Parameters
    ----------
    data_dir : str
        Directory containing per-patient ``.psv`` files.
    max_patients : int or None
        If set, load at most this many files (useful for quick debugging).

    Returns
    -------
    pd.DataFrame
        Concatenated hourly records for every patient, with a
        ``patient_id`` column derived from each filename.

    Raises
    ------
    FileNotFoundError
        If *data_dir* does not exist or contains no ``.psv`` files.
    """
    data_path = Path(data_dir)
    if not data_path.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    psv_files = sorted(data_path.glob("*.psv"))
    if not psv_files:
        raise FileNotFoundError(f"No .psv files found in {data_path}")

    if max_patients is not None:
        psv_files = psv_files[:max_patients]

    logger.info("Loading %d patient files from %s", len(psv_files), data_path)

    frames: list[pd.DataFrame] = []
    for fp in tqdm(psv_files, desc="Loading patients", unit="file"):
        try:
            frames.append(load_single_psv(fp))
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
            logger.warning("Skipping %s: %s", fp.name, exc)

    if not frames:
        raise RuntimeError("No patient files could be loaded successfully.")

    combined = pd.concat(frames, ignore_index=True)
    logger.info(
        "Loaded %d rows × %d cols for %d patients",
        len(combined),
        combined.shape[1],
        combined["patient_id"].nunique(),
    )
    return combined


# ---------------------------------------------------------------------------
# 3. Aggregate hourly → patient-level
# ---------------------------------------------------------------------------
def aggregate_to_patient_level(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse hourly rows to one row per patient via summary statistics.

    For every vital and lab feature the function computes **mean, std,
    min, max** (columns suffixed ``_mean``, ``_std``, ``_min``, ``_max``).

    * ``SepsisLabel``: **max** per patient (any positive hour → positive).
    * ``Age``, ``Gender``: first observed value retained as-is.
    * ``ICULOS``: **max** (length of stay).
    * ``Unit1``, ``Unit2``, ``HospAdmTime``: dropped.

    Parameters
    ----------
    df : pd.DataFrame
        Hourly-level data with a ``patient_id`` column.

    Returns
    -------
    pd.DataFrame
        One row per patient with aggregated features and a ``patient_id``
        index reset as a regular column.
    """
    if "patient_id" not in df.columns:
        raise ValueError("DataFrame must contain a 'patient_id' column.")

    numeric_features = [
        c for c in VITAL_FEATURES + LAB_FEATURES if c in df.columns
    ]

    # Build aggregation dictionary
    agg_dict: dict = {}
    for feat in numeric_features:
        agg_dict[feat] = ["mean", "std", "min", "max"]

    if TARGET in df.columns:
        agg_dict[TARGET] = "max"
    if "Age" in df.columns:
        agg_dict["Age"] = "first"
    if "Gender" in df.columns:
        agg_dict["Gender"] = "first"
    if "ICULOS" in df.columns:
        agg_dict["ICULOS"] = "max"

    grouped = df.groupby("patient_id").agg(agg_dict)

    # Flatten multi-level column names
    new_cols: list[str] = []
    for col_tuple in grouped.columns:
        if isinstance(col_tuple, tuple):
            top, agg_name = col_tuple
            # For columns with a single aggregation keep name clean
            if agg_name in ("max", "first") and top in (
                TARGET, "Age", "Gender", "ICULOS",
            ):
                new_cols.append(top)
            else:
                new_cols.append(f"{top}_{agg_name}")
        else:
            new_cols.append(col_tuple)
    grouped.columns = new_cols

    grouped = grouped.reset_index()

    logger.info(
        "Aggregated to %d patients × %d features",
        len(grouped),
        grouped.shape[1],
    )
    return grouped


# ---------------------------------------------------------------------------
# 4. Assign temporal phases for drift simulation
# ---------------------------------------------------------------------------
def assign_temporal_phases(
    df: pd.DataFrame,
    phase1_frac: float = 0.40,
    phase2_frac: float = 0.25,
) -> pd.DataFrame:
    """Label each patient row with a temporal **phase** for drift simulation.

    Patients are sorted by ``patient_id`` (simulating arrival order) and
    then split into three contiguous phases:

    * ``phase1`` — first *phase1_frac* of patients (reference/training).
    * ``phase2`` — next *phase2_frac* (gradual drift).
    * ``phase3`` — remainder (severe drift).

    Parameters
    ----------
    df : pd.DataFrame
        Patient-level DataFrame (must contain ``patient_id``).
    phase1_frac : float
        Fraction of patients in phase 1 (default 0.40).
    phase2_frac : float
        Fraction of patients in phase 2 (default 0.25).

    Returns
    -------
    pd.DataFrame
        Copy of *df* with an added ``phase`` column.
    """
    if phase1_frac + phase2_frac > 1.0:
        raise ValueError(
            "phase1_frac + phase2_frac must be ≤ 1.0; "
            f"got {phase1_frac} + {phase2_frac} = {phase1_frac + phase2_frac}"
        )

    df = df.sort_values("patient_id").reset_index(drop=True)
    n = len(df)
    cut1 = int(n * phase1_frac)
    cut2 = int(n * (phase1_frac + phase2_frac))

    phases = np.empty(n, dtype=object)
    phases[:cut1] = "phase1"
    phases[cut1:cut2] = "phase2"
    phases[cut2:] = "phase3"

    df["phase"] = phases

    for phase_name in ("phase1", "phase2", "phase3"):
        count = int((phases == phase_name).sum())
        logger.info("Phase %-6s : %d patients (%.1f%%)", phase_name, count, 100 * count / n)

    return df


# ---------------------------------------------------------------------------
# 5. Synthetic dataset fallback
# ---------------------------------------------------------------------------

# Typical ICU ranges: (mean, std) — used only for synthetic generation
_VITAL_RANGES: dict[str, tuple[float, float]] = {
    "HR": (82.0, 18.0),
    "O2Sat": (96.5, 2.5),
    "Temp": (37.0, 0.7),
    "SBP": (122.0, 22.0),
    "MAP": (80.0, 14.0),
    "DBP": (62.0, 12.0),
    "Resp": (19.0, 5.0),
    "EtCO2": (33.0, 5.0),
}

_LAB_RANGES: dict[str, tuple[float, float]] = {
    "BaseExcess": (0.0, 4.0),
    "HCO3": (24.0, 4.0),
    "FiO2": (0.45, 0.15),
    "pH": (7.38, 0.06),
    "PaCO2": (40.0, 6.0),
    "SaO2": (96.0, 3.0),
    "AST": (35.0, 30.0),
    "BUN": (22.0, 14.0),
    "Alkalinephos": (80.0, 35.0),
    "Calcium": (8.8, 0.8),
    "Chloride": (103.0, 5.0),
    "Creatinine": (1.2, 0.9),
    "Glucose": (130.0, 45.0),
    "Lactate": (1.8, 1.2),
    "Magnesium": (2.0, 0.4),
    "Phosphate": (3.5, 1.0),
    "Potassium": (4.2, 0.6),
    "Hct": (33.0, 6.0),
    "Hgb": (11.0, 2.0),
    "PTT": (32.0, 12.0),
    "WBC": (11.0, 5.5),
    "Fibrinogen": (280.0, 90.0),
    "Platelets": (210.0, 85.0),
    "TroponinI": (0.08, 0.20),
    "Bilirubin_total": (1.2, 1.5),
}

# Fraction of values that should be NaN for each lab (mimics real sparsity)
_LAB_MISSING_FRAC: float = 0.70
_VITAL_MISSING_FRAC: float = 0.05


def generate_synthetic_dataset(
    n_patients: int = 2000,
    n_hours_range: tuple[int, int] = (12, 72),
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic dataset that mimics PhysioNet 2019 structure.

    This is a **fallback** for environments where the real data is
    unavailable.  Values are drawn from normal distributions centred on
    typical ICU ranges with realistic missingness patterns.

    Sepsis prevalence is approximately **13 %** of patients.

    Parameters
    ----------
    n_patients : int
        Number of synthetic patients to generate.
    n_hours_range : tuple[int, int]
        (min_hours, max_hours) for the length-of-stay uniform draw.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Hourly-level synthetic data with all standard columns plus
        ``patient_id``.
    """
    rng = np.random.default_rng(seed)
    logger.info(
        "Generating synthetic dataset: %d patients, hours ∈ %s",
        n_patients,
        n_hours_range,
    )

    all_rows: list[dict] = []

    # Decide which patients get sepsis (~13 %)
    sepsis_flags = rng.random(n_patients) < 0.13

    for i in range(n_patients):
        pid = f"p{i:06d}"
        n_hours = int(rng.integers(n_hours_range[0], n_hours_range[1] + 1))
        has_sepsis = bool(sepsis_flags[i])

        # If sepsis, onset somewhere in the last third of stay
        if has_sepsis:
            onset_hour = int(rng.integers(max(1, int(n_hours * 0.66)), n_hours))
        else:
            onset_hour = n_hours + 999  # never

        age = float(rng.integers(18, 95))
        gender = int(rng.integers(0, 2))

        for h in range(1, n_hours + 1):
            row: dict = {"patient_id": pid, "ICULOS": h}

            # Vitals
            for feat, (mu, sigma) in _VITAL_RANGES.items():
                if rng.random() < _VITAL_MISSING_FRAC:
                    row[feat] = np.nan
                else:
                    val = rng.normal(mu, sigma)
                    # Physiological clamps
                    if feat == "O2Sat":
                        val = np.clip(val, 50.0, 100.0)
                    elif feat == "HR":
                        val = max(20.0, val)
                    elif feat in ("SBP", "MAP", "DBP"):
                        val = max(30.0, val)
                    elif feat == "Resp":
                        val = max(4.0, val)
                    elif feat == "Temp":
                        val = np.clip(val, 33.0, 42.0)
                    row[feat] = round(val, 1)

            # Labs — mostly missing
            for feat, (mu, sigma) in _LAB_RANGES.items():
                if rng.random() < _LAB_MISSING_FRAC:
                    row[feat] = np.nan
                else:
                    val = rng.normal(mu, sigma)
                    # Prevent negatives for strictly-positive labs
                    if feat in (
                        "AST", "BUN", "Alkalinephos", "Calcium", "Chloride",
                        "Creatinine", "Glucose", "Lactate", "Magnesium",
                        "Phosphate", "Potassium", "Hct", "Hgb", "PTT",
                        "WBC", "Fibrinogen", "Platelets", "TroponinI",
                        "Bilirubin_total", "FiO2",
                    ):
                        val = max(0.01, val)
                    if feat == "pH":
                        val = np.clip(val, 6.8, 7.8)
                    row[feat] = round(val, 2)

            # Sepsis-induced perturbation (after onset)
            if has_sepsis and h >= onset_hour:
                if "HR" in row and not np.isnan(row.get("HR", np.nan)):
                    row["HR"] = round(row["HR"] + rng.normal(15, 5), 1)
                if "Temp" in row and not np.isnan(row.get("Temp", np.nan)):
                    row["Temp"] = round(min(42.0, row["Temp"] + rng.normal(1.0, 0.3)), 1)
                if "O2Sat" in row and not np.isnan(row.get("O2Sat", np.nan)):
                    row["O2Sat"] = round(max(50.0, row["O2Sat"] - rng.normal(4, 2)), 1)
                if "WBC" in row and not np.isnan(row.get("WBC", np.nan)):
                    row["WBC"] = round(row["WBC"] * rng.uniform(1.3, 1.8), 2)
                if "Lactate" in row and not np.isnan(row.get("Lactate", np.nan)):
                    row["Lactate"] = round(row["Lactate"] * rng.uniform(1.5, 2.5), 2)

            # Demographics
            row["Age"] = age
            row["Gender"] = gender
            row["Unit1"] = int(rng.integers(0, 2))
            row["Unit2"] = 1 - row["Unit1"]
            row["HospAdmTime"] = round(float(rng.normal(-50, 30)), 1)

            # Target
            row[TARGET] = 1 if (has_sepsis and h >= onset_hour) else 0

            all_rows.append(row)

    synth_df = pd.DataFrame(all_rows)

    # Reorder columns to match real data layout
    ordered_cols = (
        VITAL_FEATURES + LAB_FEATURES + DEMO_FEATURES + [TARGET, "patient_id"]
    )
    ordered_cols = [c for c in ordered_cols if c in synth_df.columns]
    synth_df = synth_df[ordered_cols]

    n_sepsis = int(sepsis_flags.sum())
    logger.info(
        "Synthetic dataset: %d rows, %d patients (%d sepsis = %.1f%%)",
        len(synth_df),
        n_patients,
        n_sepsis,
        100 * n_sepsis / n_patients,
    )
    return synth_df
