"""Data layer: dataset loading, validation, preprocessing and rubrics."""

from src.data.beetle_loader import load_beetle
from src.data.dataset_loader import (
    REQUIRED_COLUMNS,
    DatasetMetadata,
    empty_standard_frame,
    normalize_labels,
)
from src.data.dataset_validator import frame_summary, normalize_frame, validate_frame
from src.data.preprocessing import (
    combine_datasets,
    preprocess_dataset,
    split_frame,
)
from src.data.rubrics import (
    BUILTIN_RUBRICS,
    auto_rubric_from_question,
    find_builtin_rubric,
    normalize_rubric,
)
from src.data.scientbank_loader import load_scientbank

__all__ = [
    "load_beetle",
    "load_scientbank",
    "REQUIRED_COLUMNS",
    "DatasetMetadata",
    "empty_standard_frame",
    "normalize_labels",
    "frame_summary",
    "normalize_frame",
    "validate_frame",
    "combine_datasets",
    "preprocess_dataset",
    "split_frame",
    "BUILTIN_RUBRICS",
    "auto_rubric_from_question",
    "find_builtin_rubric",
    "normalize_rubric",
]