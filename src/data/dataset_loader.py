"""Public dataset loaders for the evaluation project.

The system does NOT assume a single schema: every loader normalizes records
into one standard DataFrame shape:

    dataset | question | student_answer | reference_answer | label | score | source_file

- ``label``   : "correct" | "incorrect" (usable for classifier training)
- ``score``   : optional human score (usable for scoring experiments)
- ``reference_answer``: optional gold/reference answer

NOTE: No dataset file is fabricated. If a dataset is not placed in
``data/raw`` (licensing/access), the loader raises ``DatasetError`` with clear
instructions, and ``scripts/download_data.py`` documents how to obtain it.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from app.core.exceptions import DatasetError

REQUIRED_COLUMNS = ["dataset", "question", "student_answer"]


@dataclass
class DatasetMetadata:
    name: str
    license_note: str
    version: str = "1.0"


def empty_standard_frame(n_rows: int = 0, dataset_name: str = "unknown") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset": [dataset_name] * n_rows,
            "question": [None] * n_rows,
            "student_answer": [None] * n_rows,
            "reference_answer": [None] * n_rows,
            "label": [None] * n_rows,
            "score": [None] * n_rows,
            "source_file": [None] * n_rows,
        }
    )


CORRECT_LABELS = {
    "correct",
    "right",
    "good",
    "1.0",
    "1",
    "true",
    "a",
    # SciEntsBank / BEETLE style labels.
    "mostly correct",
    "mostly_correct",
}

# The standardized label space is binary (correct/incorrect). Partially-correct
# and mostly-incorrect responses collapse to "incorrect" for classifier
# training, which is a deliberate loss of information. Note that BEETLE's
# `partially_correct` used to be missing from this map, which silently turned
# those rows into unlabeled data instead.
INCORRECT_LABELS = {
    "incorrect",
    "wrong",
    "bad",
    "0.0",
    "0",
    "false",
    "contradictory",
    "irrelevant",
    "partial",
    "partially correct",
    "partially_correct",
    "mostly incorrect",
    "mostly_incorrect",
}


def normalize_labels(labels) -> Optional[str]:
    """Map arbitrary dataset labels to correct/incorrect (None if not derivable)."""
    if labels is None or (isinstance(labels, float) and pd.isna(labels)):
        return None
    if isinstance(labels, (list, tuple)):
        # Label lists appear in SciEntsBank (e.g. ["correct","partial"])
        return "correct" if "correct" in labels else "incorrect"
    val = str(labels).strip().lower()
    if val in CORRECT_LABELS:
        return "correct"
    if val in INCORRECT_LABELS:
        return "incorrect"
    return None


def restore_python_none(frame: pd.DataFrame) -> pd.DataFrame:
    """Return ``frame`` with missing values represented as Python ``None``.

    pandas 3.x coerces the ``None``s that mean "no label / no reference / no
    score" into ``NaN`` when a frame is built from heterogeneous dict rows
    (its new inferred ``str`` dtype also turns ``None`` back into ``NaN`` on
    assignment). The documented standard-frame contract (and downstream
    ``is None`` checks) expects ``None`` for "not provided", so restore it by
    forcing any column that has missing values to ``object`` before writing
    ``None``. Columns without missing values keep their dtype.
    """
    for col in frame.columns:
        missing = frame[col].isna()
        if not missing.any():
            continue
        if frame[col].dtype != object:
            frame[col] = frame[col].astype(object)
        frame.loc[missing, col] = None
    return frame


def require_raw_file(path: str, instructions: str) -> Path:
    p = Path(path)
    if not p.exists():
        raise DatasetError(
            f"Dataset file not found at {p}.\n{instructions}"
        )
    return p