"""ASAP-AES loader (CSV from the ASAP Automated Essay Scoring competition).

This dataset grades full essays, not short answers. It is included so the
scoring experiments can compare against human scores.

Obtain: Kaggle "ASAP-AES" (https://www.kaggle.com/c/asap-aes) - requires a
Kaggle account (license: competition data, research use). Place the CSV (and
optional prompts file) under data/raw/asap/ and point the loader at them.
"""

from typing import Dict, Optional

import pandas as pd

from app.core.exceptions import DatasetError
from src.data.dataset_loader import require_raw_file, restore_python_none

PROMPT_COLUMNS = ("prompt_id", "essay_set", "set")
ESSAY_COLUMNS = ("essay", "text", "answer")
SCORE_COLUMNS = ("domain1_score", "score", "human_score")

# The ASAP prompts file keys prompts differently from the essay file
# ("PromptID" vs "essay_set"), so the join column is detected independently.
PROMPT_KEY_CANDIDATES = ("prompt_id", "PromptID", "essay_set", "set", "id")
PROMPT_TEXT_CANDIDATES = ("description", "prompt", "Prompt", "text", "essay_set_description")


def _column(df: pd.DataFrame, candidates, label: str) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    raise DatasetError(
        f"ASAP dataset is missing a '{label}' column. Expected one of: {candidates}."
    )


def _optional_column(df: pd.DataFrame, candidates) -> Optional[str]:
    """First candidate column present, matched case-insensitively.

    ASAP releases are inconsistent about capitalisation (`PromptID` vs
    `prompt_id`, `Description` vs `description`). A case-sensitive lookup
    silently selected the wrong column - it picked the `Prompt` label instead of
    the `Description` body - and produced plausible but wrong question text.
    """
    by_lower = {str(col).lower(): col for col in df.columns}
    for col in candidates:
        if col in df.columns:
            return col
        actual = by_lower.get(col.lower())
        if actual is not None:
            return actual
    return None


def load_asap(
    path: str,
    prompts_path: Optional[str] = None,
    dataset_name: str = "asap-aes",
) -> pd.DataFrame:
    src = require_raw_file(
        path,
        "Download the ASAP-AES CSV from Kaggle, place it under data/raw/asap/, "
        "and pass the file path here.",
    )
    df = pd.read_csv(src)

    prompt_col = _column(df, PROMPT_COLUMNS, "prompt")
    essay_col = _column(df, ESSAY_COLUMNS, "essay")
    score_col = _optional_column(df, SCORE_COLUMNS)

    prompts: Dict[object, str] = {}
    prompts_join_note: Optional[str] = None
    if prompts_path:
        prompts_file = require_raw_file(
            prompts_path,
            "Optional prompts CSV must exist to enrich questions.",
        )
        prompts_df = pd.read_csv(prompts_file)
        key_col = _optional_column(prompts_df, PROMPT_KEY_CANDIDATES)
        text_col = _optional_column(prompts_df, PROMPT_TEXT_CANDIDATES)
        if key_col and text_col:
            for _, row in prompts_df.iterrows():
                key = row.get(key_col)
                text = row.get(text_col)
                if key is not None and text is not None and str(text).strip():
                    prompts[key] = str(text).strip()
        else:
            prompts_join_note = (
                f"Prompts file {prompts_file.name} has no recognisable key/text "
                f"column (columns: {list(prompts_df.columns)}); questions were "
                "left as placeholders."
            )

    rows = []
    placeholder_count = 0
    for _, row in df.iterrows():
        essay = str(row[essay_col])
        if not essay or not essay.strip():
            continue
        pid = row.get(prompt_col)
        question = prompts.get(pid) if pid is not None else None
        if question is None:
            # Placeholder is marked as such so it can never be mistaken for the
            # real prompt text during training or in a report.
            placeholder_count += 1
            question = (
                f"[prompt text unavailable for essay set {pid}] "
                "Write an essay responding to the assigned ASAP prompt."
            )
        score = None
        if score_col is not None:
            val = row.get(score_col)
            if pd.notna(val):
                score = float(val)
        rows.append(
            {
                "dataset": dataset_name,
                "question": question,
                "student_answer": essay.strip(),
                "reference_answer": str(row.get("reference_answer") or "").strip() or None,
                # ASAP-AES ships human *scores*, not correct/incorrect labels,
                # so `label` stays None and this dataset can only be used for
                # regression/scoring experiments, never classifier training.
                "label": None,
                "score": score,
                "source_file": src.name,
            }
        )

    if not rows:
        raise DatasetError("ASAP dataset produced no usable rows.")

    frame = restore_python_none(pd.DataFrame(rows))
    # Surfaced so callers/report generators can state the caveat rather than
    # implying every question was the real prompt.
    frame.attrs["prompt_placeholders"] = placeholder_count
    if prompts_join_note:
        frame.attrs["prompts_note"] = prompts_join_note
    return frame