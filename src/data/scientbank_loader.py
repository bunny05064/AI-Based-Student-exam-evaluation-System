"""SciEntsBank loader (.json files from the NIST/HUSE AS-2 shared task).

The SciEntsBank corpus ships as JSON documents (e.g. NAEP.json,
Categories.json, All.json ...). Field names vary slightly by release, so the
loader uses ordered, configurable aliases. If a required field cannot be
resolved it raises DatasetError instead of guessing.

Obtain the data: https://github.com/openedu-data/... or the HUSE task release;
place the JSON file(s) under data/raw/scientbank/ and point the loader at them.
"""

import json
from pathlib import Path
from typing import List, Optional

import pandas as pd

from app.core.exceptions import DatasetError
from src.data.dataset_loader import (
    normalize_labels,
    require_raw_file,
    restore_python_none,
)

QUESTION_KEYS = ["question", "Q", "text", "prompt", "Question"]
REFERENCE_KEYS = [
    "reference_answer",
    "reference",
    "gold_answer",
    "expected_answer",
    "correct_answer",
    "EA",
    "RF",
]

# Candidate *student* answer fields and the label implied by the field name.
#
# NOTE: `correct_answer` is deliberately absent. In SciEntsBank it holds the
# gold answer, not something a student wrote. Treating it as a labelled
# "correct" student answer leaks the reference into the training data and
# inflates every reported metric. It is collected as REFERENCE_KEYS instead.
#
# "partial" maps to "incorrect" for the binary correct/incorrect label space,
# which is a real loss of information; the standardized frame is binary.
STUDENT_ANSWER_FIELDS = [
    ("incorrect_answer", "incorrect"),
    ("partially_correct_answer", "partial"),
    ("student_answer", None),
    ("SA", None),
]


def _pick(record: dict, keys: List[str]) -> Optional[str]:
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, list) and value and isinstance(value[0], str):
            return " ".join(str(v) for v in value)
    return None


def _student_answers(record: dict) -> List[tuple]:
    """Collect (candidate_answer, label) pairs from every student-answer field.

    All aliases are read because a record can legitimately carry distinct
    answers in different fields (e.g. `incorrect_answer` *and* `student_answer`).
    Duplicates are removed on the answer text, which is what actually happens
    when a release carries the same response under both `student_answer` and
    `SA` - scanning every alias naively double-counted those rows and biased
    the class balance.
    """
    seen = set()
    pairs: List[tuple] = []
    for key, label in STUDENT_ANSWER_FIELDS:
        if key not in record:
            continue
        value = record[key]
        candidates: List[tuple] = []
        if isinstance(value, str) and value.strip():
            candidates.append((value.strip(), label))
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str) and item.strip():
                    candidates.append((item.strip(), label))
                elif isinstance(item, dict):
                    txt = _pick(item, ["answer", "text", "student_answer", "SA"])
                    lab = normalize_labels(item.get("label")) or label
                    if txt:
                        candidates.append((txt, lab))
        for text, lab in candidates:
            if text in seen:
                continue
            seen.add(text)
            pairs.append((text, lab))
    return pairs


def load_scientbank(path: str, dataset_name: str = "scientbank") -> pd.DataFrame:
    """Load one JSON file or a directory of JSON files."""
    src = require_raw_file(
        path,
        "Download a SciEntsBank release, place its JSON file(s) under "
        "data/raw/scientbank/, and pass the file or directory path here.",
    )
    files: List[Path] = [src] if src.is_file() else sorted(src.glob("*.json"))
    if not files and src.is_dir():
        files = sorted(src.glob("**/*.json"))
    if not files:
        raise DatasetError(f"No .json files found in {path}.")

    rows = []
    skipped_no_question = 0
    for file in files:
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise DatasetError(f"Invalid JSON in {file}: {exc}") from exc
        if isinstance(data, list):
            entries = data
        elif isinstance(data, dict):
            entries = data.get("entries") or data.get("questions") or []
        else:
            entries = []
        for record in entries:
            if not isinstance(record, dict):
                continue
            question = _pick(record, QUESTION_KEYS)
            if not question:
                skipped_no_question += 1
                continue
            ref = _pick(record, REFERENCE_KEYS)
            for answer, label in _student_answers(record):
                rows.append(
                    {
                        "dataset": dataset_name,
                        "question": question,
                        "student_answer": answer,
                        "reference_answer": ref,
                        "label": normalize_labels(label),
                        "score": None,
                        "source_file": file.name,
                    }
                )

    if not rows:
        # Report what was actually seen so the schema gap is actionable
        # instead of looking like an empty download.
        seen = set()
        for file in files:
            try:
                data = json.loads(file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            entries = data if isinstance(data, list) else (
                data.get("entries") or data.get("questions") or []
                if isinstance(data, dict) else []
            )
            for record in entries:
                if isinstance(record, dict):
                    seen.update(record.keys())
        raise DatasetError(
            f"No usable records extracted from {src}. "
            f"Records parsed: {sum(1 for _ in files)} file(s); "
            f"{skipped_no_question} record(s) had no recognisable question. "
            f"Keys seen: {sorted(seen) or 'none'}. "
            "Expected one of "
            f"{QUESTION_KEYS} for the question and one of "
            f"{[k for k, _ in STUDENT_ANSWER_FIELDS]} for a student answer."
        )
    return restore_python_none(pd.DataFrame(rows))