"""Metrics for the ML components (classification + regression + correlation).

Classification: accuracy, precision, recall, F1, confusion matrix.
Scoring: MAE, RMSE, Pearson/Spearman correlation vs human labels.
"""

from typing import List, Sequence

import numpy as np

from src.ml.similarity import pearson_correlation, spearman_correlation


def classification_metrics(
    y_true: Sequence[str], y_pred: Sequence[str]
) -> dict:
    """Compute accuracy/precision/recall/F1 treating 'correct' as positive."""
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        precision_recall_fscore_support,
    )

    y_true_l = list(y_true)
    y_pred_l = list(y_pred)
    if not y_true_l:
        raise ValueError("y_true must not be empty.")

    acc = accuracy_score(y_true_l, y_pred_l)
    # Index of positive class if it exists.
    pos_label = "correct" if "correct" in set(y_true_l) else y_pred_l[0]
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_l, y_pred_l, average="binary", pos_label=pos_label, zero_division=0
    )
    cm = confusion_matrix(y_true_l, y_pred_l).tolist()
    return {
        "accuracy": round(float(acc), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "confusion_matrix": cm,
    }


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> dict:
    y_true_a = np.asarray(y_true, dtype=float)
    y_pred_a = np.asarray(y_pred, dtype=float)
    errors = y_pred_a - y_true_a
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "pearson": round(pearson_correlation(y_true, y_pred), 4),
        "spearman": round(spearman_correlation(y_true, y_pred), 4),
    }


def correlation_metrics(similarities: Sequence[float], human_scores: Sequence[float]) -> dict:
    return {
        "pearson": round(pearson_correlation(similarities, human_scores), 4),
        "spearman": round(spearman_correlation(similarities, human_scores), 4),
    }


def format_confusion_matrix(cm: List[List[int]], labels: Sequence[str]) -> str:
    header = "      " + "  ".join(f"{l:>6}" for l in labels)
    lines = [header]
    for row, label in zip(cm, labels):
        lines.append(f"{label:>6}" + "".join(f" {v:>6}" for v in row))
    return "\n".join(lines)