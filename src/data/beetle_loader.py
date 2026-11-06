"""BEETLE loader (XML from the BEETLEeval shared task on answer correctness).

BEETLEeval releases ship XML in which each graded answer is one record. Two
shapes are handled:

* element-per-answer, e.g. ``<student-answer label="correct">text</student-answer>``
* attribute-per-answer, e.g. ``<Answer answer="..." label="correct" .../>``

Attribute records are common in the released "Answers" XML; the previous
version only looked at child elements, so those files parsed to zero rows and
raised with no hint about why.

Every accepted field name is declared in the constants below, so a release
that uses different names produces an actionable error listing what was found
rather than a silently empty frame.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from app.core.exceptions import DatasetError
from src.data.dataset_loader import (
    normalize_labels,
    require_raw_file,
    restore_python_none,
)

# Question text: checked as a child element, then as an attribute.
QUESTION_TAGS = ("question", "Q", "prompt", "stem", "questionText")
QUESTION_ATTRS = ("question", "prompt", "prompt_text", "text")

# Reference / gold answer.
REFERENCE_TAGS = ("reference", "reference-answer", "referenceAnswer", "RF", "gold")
REFERENCE_ATTRS = ("reference", "reference_answer", "referenceAnswer", "gold", "RF")

# Student answer: the first alias present wins, so one record never yields
# several copies of the same answer.
ANSWER_TAGS = ("student-answer", "student_answer", "studentAnswer", "SA", "answer")
ANSWER_ATTRS = ("answer", "student_answer", "studentAnswer", "SA", "text")

LABEL_ATTRS = ("label", "grade", "correctness", "correct", "type", "category")
LABEL_TAGS = ("label", "grade", "correctness")

# Elements that wrap a list of answers rather than being one.
CONTAINER_TAGS = ("answers", "answer_list", "instances", "records", "data", "root")


def _text_of(element: Optional[ET.Element]) -> Optional[str]:
    if element is None:
        return None
    text = " ".join((element.text or "").strip().split())
    if text:
        return text
    # Fall back to nested text for <student-answer>text<b>!</b></student-answer>.
    nested = " ".join(part.strip() for part in element.itertext() if part and part.strip())
    return " ".join(nested.split()) or None


def _attr_text(element: ET.Element, names) -> Optional[str]:
    for name in names:
        value = element.get(name)
        if value and value.strip():
            return " ".join(value.split())
    return None


def _first_text(root: ET.Element, tags, attrs) -> Optional[str]:
    for tag in tags:
        for found in root.iter(tag):
            text = _text_of(found)
            if text:
                return text
    return _attr_text(root, attrs)


def _label_of(element: ET.Element) -> Optional[str]:
    for attr in LABEL_ATTRS:
        value = element.get(attr)
        if value:
            return normalize_labels(value)
    for tag in LABEL_TAGS:
        found = element.find(f".//{tag}")
        if found is not None:
            text = _text_of(found)
            if text:
                return normalize_labels(text)
    return None


def _iter_records(root: ET.Element) -> List[ET.Element]:
    """Yield the elements that plausibly represent one graded answer."""
    records: List[ET.Element] = []
    for element in root.iter():
        if element.tag in CONTAINER_TAGS:
            continue
        if element.tag in ANSWER_TAGS:
            records.append(element)
            continue
        # Attribute-style record: has an answer-ish attribute.
        if any(element.get(name) for name in ANSWER_ATTRS):
            records.append(element)
    return records


def _answer_of(record: ET.Element) -> Optional[str]:
    if record.tag in ANSWER_TAGS:
        text = _text_of(record)
        if text:
            return text
    return _attr_text(record, ANSWER_ATTRS)


def _resolve_question(record: ET.Element, root_question: Optional[str]) -> Optional[str]:
    """Per-record question, falling back to the document-level one."""
    if record.tag not in CONTAINER_TAGS:
        own = _attr_text(record, QUESTION_ATTRS)
        if own:
            return own
    return root_question


def _xml_files(src: Path) -> List[Path]:
    if src.is_file():
        return [src]
    files = sorted(src.glob("*.xml"))
    if not files:
        files = sorted(src.glob("**/*.xml"))
    return files


def load_beetle(path: str, dataset_name: str = "beetle") -> pd.DataFrame:
    src = require_raw_file(
        path,
        "Download the BEETLEeval dataset, place its XML files under "
        "data/raw/beetle/, and pass the file or directory path here.",
    )
    files = _xml_files(src)
    if not files:
        raise DatasetError(f"No .xml files found in {path}.")

    rows: List[Dict] = []
    seen_tags = set()
    for file in files:
        try:
            tree = ET.parse(file)
        except ET.ParseError as exc:
            raise DatasetError(f"Invalid XML in {file}: {exc}") from exc
        root = tree.getroot()
        seen_tags.update(el.tag for el in root.iter())

        root_question = _first_text(root, QUESTION_TAGS, QUESTION_ATTRS)
        root_reference = _first_text(root, REFERENCE_TAGS, REFERENCE_ATTRS)

        for record in _iter_records(root):
            answer = _answer_of(record)
            if not answer:
                continue
            question = _resolve_question(record, root_question)
            if not question:
                continue
            rows.append(
                {
                    "dataset": dataset_name,
                    "question": question,
                    "student_answer": answer,
                    "reference_answer": _attr_text(record, REFERENCE_ATTRS) or root_reference,
                    "label": _label_of(record),
                    "score": None,
                    "source_file": file.name,
                }
            )

    if not rows:
        raise DatasetError(
            f"No usable records extracted from {src}. "
            f"XML tags present: {sorted(seen_tags) or 'none'}. "
            f"Expected a question in any of {QUESTION_TAGS + QUESTION_ATTRS} and a "
            f"student answer in any of {ANSWER_TAGS + ANSWER_ATTRS}. "
            "If this release uses different names, add them to the loader "
            "constants rather than assuming an empty result means an empty "
            "download."
        )

    frame = restore_python_none(pd.DataFrame(rows))
    # One (question, answer) pair must not appear twice: attribute-style records
    # can match both the ANSWER_TAGS and ANSWER_ATTRS paths.
    before = len(frame)
    frame = frame.drop_duplicates(subset=["question", "student_answer"], keep="first")
    if len(frame) < before:
        frame.attrs["duplicate_rows_removed"] = before - len(frame)
    return frame.reset_index(drop=True)


__all__ = ["load_beetle"]
