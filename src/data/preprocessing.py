"""Reproducible dataset preprocessing.

Pipeline (scripts/preprocess_data.py drives this):
1. Validate raw frame
2. Clean / normalize text (keeps math, no aggressive stopword removal)
3. Drop empty rows, deduplicate, normalize labels
4. Train/validation/test split - BEFORE any TF-IDF fit / embedding build
   (prevents data leakage; the split is recorded in the report)
5. Write processed CSVs + a preprocessing report (reproducibility)

Cleaning notes:
- Unicode NFKC, whitespace collapse, quote/space-artifact fixes.
- Mathematical expressions like "F=ma" are preserved (not token-stripped).
- Stopwords are NOT aggressively removed (transformers need context).
"""

import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from sklearn.model_selection import train_test_split

from app.core.exceptions import DatasetError
from app.utils.text import clean_text
from src.data.dataset_validator import frame_summary, normalize_frame, validate_frame

DEFAULT_VAL_FRAC = 0.15
DEFAULT_TEST_FRAC = 0.15


def _clean_column(series: pd.Series) -> pd.Series:
    return series.map(lambda v: clean_text("" if pd.isna(v) else v) if isinstance(v, str) else v)


def _stratification_skip_reason(frame: pd.DataFrame, column: Optional[str]) -> str:
    """Explain, in the report, why a stratified split was not possible."""
    if column is None or column not in frame.columns:
        return f"No stratify column available (requested: {column!r})."
    values = frame[column]
    if not values.notna().all():
        return f"Column {column!r} contains missing values."
    if values.nunique() < 2:
        return f"Column {column!r} has a single class."
    smallest = int(values.value_counts().min())
    return (
        f"Column {column!r} has classes with fewer than 2 members "
        f"(smallest class = {smallest}); sklearn cannot stratify it."
    )


def _usable_strata(frame: pd.DataFrame, column: Optional[str]) -> Optional[pd.Series]:
    """Return the column if every class has at least 2 members, else None.

    sklearn raises when a class has a single member, which is common in small
    answer-key datasets. Falling back to an unstratified split keeps
    preprocessing working instead of crashing.
    """
    if column is None or column not in frame.columns:
        return None
    values = frame[column]
    if not values.notna().all():
        return None
    if values.nunique() < 2:
        return None
    smallest_class = int(values.value_counts().min())
    if smallest_class < 2:
        return None
    return values


def split_frame(
    frame: pd.DataFrame,
    seed: int = 42,
    val_frac: float = DEFAULT_VAL_FRAC,
    test_frac: float = DEFAULT_TEST_FRAC,
    stratify_col: Optional[str] = None,
) -> dict:
    """Stratified split when labels exist; otherwise a seeded random split.

    Never touches features - only index/rows - so no information leaks.
    """
    if val_frac + test_frac >= 1.0:
        raise DatasetError("val_frac + test_frac must be < 1.0.")

    stratify = _usable_strata(frame, stratify_col)
    train, temp = train_test_split(
        frame,
        test_size=val_frac + test_frac,
        stratify=stratify,
        random_state=seed,
    )
    stratify_val = _usable_strata(temp, stratify_col) if stratify is not None else None
    val, test = train_test_split(
        temp,
        test_size=test_frac / (val_frac + test_frac),
        stratify=stratify_val,
        random_state=seed + 1,
    )
    result = {
        "train": train.reset_index(drop=True),
        "val": val.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }
    return result


def preprocess_dataset(
    raw_frame: pd.DataFrame,
    dataset_name: str,
    processed_dir: str,
    seed: int = 42,
    val_frac: float = DEFAULT_VAL_FRAC,
    test_frac: float = DEFAULT_TEST_FRAC,
) -> dict:
    """Full per-dataset preprocessing; writes CSVs + report into processed_dir."""
    validate_frame(raw_frame)
    frame, changes = normalize_frame(raw_frame)

    frame["question"] = _clean_column(frame["question"])
    frame["student_answer"] = _clean_column(frame["student_answer"])
    if "reference_answer" in frame.columns:
        frame["reference_answer"] = _clean_column(frame["reference_answer"])

    frame = frame[(frame["question"] != "") & (frame["student_answer"] != "")]
    if len(frame) == 0:
        raise DatasetError("All rows were dropped during preprocessing.")

    stratify_col = "label" if frame["label"].notna().any() else None
    if stratify_col is None:
        frame["label"] = None

    strata = _usable_strata(frame, stratify_col)
    if strata is None:
        stratify_reason = _stratification_skip_reason(frame, stratify_col)
    else:
        stratify_reason = None

    splits = split_frame(
        frame,
        seed=seed,
        val_frac=val_frac,
        test_frac=test_frac,
        stratify_col=stratify_col,
    )

    out_dir = Path(processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report: dict = {
        "dataset": dataset_name,
        "normalization": changes,
        "summary": frame_summary(frame),
        "split_fractions": {"val": val_frac, "test": test_frac},
        "split_sizes": {
            "train": len(splits["train"]),
            "val": len(splits["val"]),
            "test": len(splits["test"]),
        },
        # Honest about what actually happened: "stratified_on" is null and a
        # reason is recorded whenever the split could not be stratified.
        "stratified_on": stratify_col if strata is not None else None,
        "stratification_skipped_reason": stratify_reason,
        "seed": seed,
        "leakage_prevention": (
            "Split performed on rows only, before any TF-IDF fit or embedding "
            "construction; test rows never influence feature extraction."
        ),
    }

    report["files"] = {}
    for split_name, split_df in splits.items():
        path = out_dir / f"{dataset_name}_{split_name}.csv"
        split_df.to_csv(path, index=False)
        report["files"][split_name] = str(path)
    full_path = out_dir / f"{dataset_name}_all.csv"
    frame.to_csv(full_path, index=False)
    report["files"]["all"] = str(full_path)

    report_path = out_dir / f"{dataset_name}_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report_file"] = str(report_path)
    return report


def combine_datasets(
    frames: Dict[str, pd.DataFrame], processed_dir: str, seed: int = 42
) -> dict:
    """Combine multiple normalized datasets into a single modeling.csv."""
    if not frames:
        raise DatasetError("No datasets provided to combine.")
    combined = pd.concat([f for f in frames.values() if len(f) > 0], ignore_index=True)
    combined = combined.drop_duplicates(subset=["question", "student_answer"], keep="first")
    combined = combined.reset_index(drop=True)

    report: dict = {
        "datasets_combined": list(frames.keys()),
        "total_rows": len(combined),
        "label_distribution": frame_summary(combined)["label_distribution"],
        "seed": seed,
        "splitting_recommendation": (
            "Run scripts/preprocess_data.py --combine with the same seed to "
            "reproduce splits; splits happen before feature engineering."
        ),
    }

    # Also produce the split modeling sets from the combined frame.
    stratify = "label" if combined["label"].notna().any() else None
    splits = split_frame(combined, seed=seed, stratify_col=stratify)
    out_dir = Path(processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_dir / "modeling.csv", index=False)
    report["files"] = {}
    for name, df in splits.items():
        combined_path = out_dir / f"modeling_{name}.csv"
        df.to_csv(combined_path, index=False)
        report["files"][name] = str(combined_path)

    (out_dir / "modeling_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report