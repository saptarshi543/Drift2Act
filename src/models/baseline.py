"""
Baseline model training and evaluation for sepsis prediction.

Provides XGBoost and Logistic Regression training with class-imbalance
handling, comprehensive evaluation (AUPRC, AUROC, Brier score), MLflow
experiment tracking, and model persistence via joblib.
"""

import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
    precision_recall_curve,
    classification_report,
)
import numpy as np
import pandas as pd
import mlflow
import joblib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def train_xgboost(
    X_train: np.ndarray | pd.DataFrame,
    y_train: np.ndarray | pd.Series,
    X_val: np.ndarray | pd.DataFrame,
    y_val: np.ndarray | pd.Series,
    n_estimators: int = 300,
    max_depth: int = 6,
    learning_rate: float = 0.03,
    random_state: int = 42,
) -> xgb.XGBClassifier:
    """Train an XGBoost classifier tuned for imbalanced sepsis prediction.

    Uses scale_pos_weight to compensate for the severe class imbalance in
    PhysioNet 2019 data (~1.8% positive rate), AUCPR as the evaluation
    metric, and early stopping to prevent overfitting.

    Args:
        X_train: Training feature matrix.
        y_train: Training binary labels (0/1).
        X_val: Validation feature matrix for early stopping.
        y_val: Validation binary labels.
        n_estimators: Maximum boosting rounds (early stopping may halt earlier).
        max_depth: Maximum tree depth.
        learning_rate: Boosting learning rate (eta).
        random_state: Random seed for reproducibility.

    Returns:
        Fitted XGBClassifier instance.
    """
    # Compute class imbalance ratio for scale_pos_weight
    y_train_arr = np.asarray(y_train)
    n_negative = int(np.sum(y_train_arr == 0))
    n_positive = int(np.sum(y_train_arr == 1))

    if n_positive == 0:
        raise ValueError("No positive samples found in y_train. Cannot train.")

    scale_pos_weight = n_negative / n_positive
    logger.info(
        "Class distribution — neg: %d, pos: %d, scale_pos_weight: %.2f",
        n_negative,
        n_positive,
        scale_pos_weight,
    )

    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        early_stopping_rounds=30,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        random_state=random_state,
        use_label_encoder=False,
    )

    logger.info("Training XGBoost (max %d rounds, early_stopping=30)…", n_estimators)

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    best_iteration = getattr(model, "best_iteration", n_estimators)
    logger.info("XGBoost training complete — best iteration: %d", best_iteration)

    # Log hyperparameters and training metadata to MLflow
    try:
        mlflow.log_params(
            {
                "xgb_n_estimators": n_estimators,
                "xgb_max_depth": max_depth,
                "xgb_learning_rate": learning_rate,
                "xgb_scale_pos_weight": round(scale_pos_weight, 4),
                "xgb_subsample": 0.8,
                "xgb_colsample_bytree": 0.8,
                "xgb_tree_method": "hist",
                "xgb_early_stopping_rounds": 30,
                "xgb_best_iteration": best_iteration,
            }
        )
    except mlflow.exceptions.MlflowException:
        logger.warning("MLflow logging skipped — no active run.")

    return model


def train_logistic_regression(
    X_train: np.ndarray | pd.DataFrame,
    y_train: np.ndarray | pd.Series,
    max_iter: int = 1000,
) -> LogisticRegression:
    """Train a Logistic Regression baseline with class-weight balancing.

    Serves as a simple, interpretable benchmark to compare against
    the XGBoost model.

    Args:
        X_train: Training feature matrix.
        y_train: Training binary labels (0/1).
        max_iter: Maximum solver iterations.

    Returns:
        Fitted LogisticRegression instance.
    """
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=max_iter,
        solver="lbfgs",
        random_state=42,
    )

    logger.info("Training Logistic Regression (max_iter=%d)…", max_iter)
    model.fit(X_train, y_train)
    logger.info("Logistic Regression training complete.")

    try:
        mlflow.log_params(
            {
                "lr_max_iter": max_iter,
                "lr_class_weight": "balanced",
                "lr_solver": "lbfgs",
            }
        )
    except mlflow.exceptions.MlflowException:
        logger.warning("MLflow logging skipped — no active run.")

    return model


def evaluate_model(
    model,
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    model_name: str = "model",
) -> dict:
    """Evaluate a binary classifier with clinical sepsis-prediction metrics.

    Computes AUPRC (primary metric for imbalanced data), AUROC, Brier
    calibration score, and the optimal threshold from the precision-recall
    curve (maximising F1).  Everything is logged to MLflow.

    Args:
        model: Fitted classifier with ``predict_proba`` method.
        X: Feature matrix.
        y: True binary labels.
        model_name: Name prefix for MLflow metric keys.

    Returns:
        Dictionary with keys: auprc, auroc, brier_score, optimal_threshold,
        classification_report.
    """
    y_arr = np.asarray(y)
    y_proba = model.predict_proba(X)[:, 1]

    # Core metrics
    auprc = average_precision_score(y_arr, y_proba)
    auroc = roc_auc_score(y_arr, y_proba)
    brier = brier_score_loss(y_arr, y_proba)

    # Optimal threshold via PR curve (maximise F1 = 2PR/(P+R))
    precision, recall, thresholds = precision_recall_curve(y_arr, y_proba)
    # precision and recall arrays are 1 element longer than thresholds
    precision_trunc = precision[:-1]
    recall_trunc = recall[:-1]

    with np.errstate(divide="ignore", invalid="ignore"):
        f1_scores = np.where(
            (precision_trunc + recall_trunc) > 0,
            2 * precision_trunc * recall_trunc / (precision_trunc + recall_trunc),
            0.0,
        )

    best_idx = int(np.argmax(f1_scores))
    optimal_threshold = float(thresholds[best_idx]) if len(thresholds) > 0 else 0.5

    # Classification report at optimal threshold
    y_pred = (y_proba >= optimal_threshold).astype(int)
    report = classification_report(y_arr, y_pred, output_dict=True)

    logger.info(
        "%s — AUPRC: %.4f | AUROC: %.4f | Brier: %.4f | Threshold: %.4f",
        model_name,
        auprc,
        auroc,
        brier,
        optimal_threshold,
    )

    metrics = {
        "auprc": auprc,
        "auroc": auroc,
        "brier_score": brier,
        "optimal_threshold": optimal_threshold,
        "classification_report": report,
    }

    # Log to MLflow
    try:
        mlflow.log_metrics(
            {
                f"{model_name}_auprc": auprc,
                f"{model_name}_auroc": auroc,
                f"{model_name}_brier_score": brier,
                f"{model_name}_optimal_threshold": optimal_threshold,
                f"{model_name}_f1_at_threshold": float(f1_scores[best_idx]),
            }
        )
    except mlflow.exceptions.MlflowException:
        logger.warning("MLflow logging skipped — no active run.")

    return metrics


def save_model(model, path: str) -> None:
    """Persist a trained model to disk via joblib.

    Creates parent directories automatically if they do not exist.

    Args:
        model: Any picklable model object.
        path: Destination file path (e.g. ``models/xgb_sepsis.joblib``).
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out)
    logger.info("Model saved to %s", out)


def load_model(path: str):
    """Load a previously saved model from disk.

    Args:
        path: Path to the joblib-serialised model file.

    Returns:
        The deserialised model object.

    Raises:
        FileNotFoundError: If the path does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Model file not found: {p}")
    model = joblib.load(p)
    logger.info("Model loaded from %s", p)
    return model
