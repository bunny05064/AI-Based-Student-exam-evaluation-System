"""Dataset validation & normalization (schema detection, missing values,
duplicates, and label normalization)."""

from typing import Tuple

import pandas as pd

from app.core.exceptions import DatasetError
from src.data.dataset_loader import REQUIRED_COLUMNS, normalize_labels

NUMERIC_COLUMNS = ("score",)


def validate_frame(df: pd.DataFrame) -> dict:
    """Static checks; returns a report dict or raises DatasetError."""
    report: dict = {"ok": True, "issues": []}
    if df is None or len(df) == 0:
        raise DatasetError("Dataset is empty - nothing to validate.")

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise DatasetError(
            f"Dataset missing required columns: {missing_cols}. "
            f"Required: {list(REQUIRED_COLUMNS)}."
        )

    empty_questions = int(df["question"].isna().sum())
    empty_answers = int(df["student_answer"].isna().sum())
    if empty_questions + empty_answers > 0:
        report["issues"].append(
            f"Missing values: {empty_questions} question, {empty_answers} answer."
        )

    dupes = int(df.duplicated(subset=["question", "student_answer"]).sum())
    if dupes:
        report["issues"].append(f"Duplicate (question, answer) rows: {dupes}.")

    invalid_labels = sum(
        1
        for l in df["label"].dropna()
        if normalize_labels(l) is None
    )
    if invalid_labels:
        report["issues"].append(f"Unrecognized label values: {invalid_labels}.")

    report.update(
        {
            "rows": len(df),
            "n_questions": int(df["question"].nunique()),
            "empty_question_rows": empty_questions,
            "empty_answer_rows": empty_answers,
            "duplicates": dupes,
            "unrecognized_labels": invalid_labels,
        }
    )
    return report


def normalize_frame(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """Apply deterministic normalizations and record what changed."""
    frame = df.copy()
    changes = {"rows_before": len(frame)}

    for col in REQUIRED_COLUMNS:
        frame[col] = frame[col].map(lambda v: "" if pd.isna(v) else str(v).strip())

    # Keep only rows with a real question AND answer.
    frame = frame[(frame["question"] != "") & (frame["student_answer"] != "")]
    changes["rows_after_dropping_empties"] = len(frame)

    # Normalize labels (mapped to None when not derivable).
    frame["label"] = frame["label"].map(normalize_labels)

    # Deduplicate exact (question, answer) pairs - keep first occurrence.
    before_dupes = len(frame)
    frame = frame.drop_duplicates(subset=["question", "student_answer"], keep="first")
    changes["duplicates_removed"] = before_dupes - len(frame)

    # Coerce numeric score column if present.
    if "score" in frame.columns:
        frame["score"] = pd.to_numeric(frame["score"], errors="coerce")

    frame = frame.reset_index(drop=True)
    return frame, changes


def frame_summary(df: pd.DataFrame) -> dict:
    labels = df["label"].dropna().value_counts().to_dict()
    score_series = df["score"].dropna() if "score" in df.columns else pd.Series(dtype=float)
    return {
        "rows": len(df),
        "datasets": df["dataset"].value_counts().to_dict(),
        "dataset_names": sorted(df["dataset"].unique().tolist()),
        "label_distribution": labels,
        "score_mean": round(float(score_series.mean()), 3) if len(score_series) else None,
        "score_min": float(score_series.min()) if len(score_series) else None,
        "score_max": float(score_series.max()) if len(score_series) else None,
    }