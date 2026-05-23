"""Drift2Act preprocessing subpackage."""

from src.preprocessing.loader import (
    VITAL_FEATURES,
    LAB_FEATURES,
    DEMO_FEATURES,
    ALL_FEATURES,
    TARGET,
    load_single_psv,
    load_physionet,
    aggregate_to_patient_level,
    assign_temporal_phases,
    generate_synthetic_dataset,
)
from src.preprocessing.imputer import (
    forward_fill_patient,
    drop_high_missing,
    knn_impute,
    scale_features,
    run_preprocessing_pipeline,
)
from src.preprocessing.windower import (
    build_temporal_windows,
    StreamingWindowIterator,
    inject_phase2_drift,
    inject_phase3_drift,
    save_drift_ground_truth,
)

__all__ = [
    # constants
    "VITAL_FEATURES",
    "LAB_FEATURES",
    "DEMO_FEATURES",
    "ALL_FEATURES",
    "TARGET",
    # loader
    "load_single_psv",
    "load_physionet",
    "aggregate_to_patient_level",
    "assign_temporal_phases",
    "generate_synthetic_dataset",
    # imputer
    "forward_fill_patient",
    "drop_high_missing",
    "knn_impute",
    "scale_features",
    "run_preprocessing_pipeline",
    # windower
    "build_temporal_windows",
    "StreamingWindowIterator",
    "inject_phase2_drift",
    "inject_phase3_drift",
    "save_drift_ground_truth",
]
