"""Single source of truth for the label space.

Previously ``src/ml/training.py`` wrote ``{"correct": 0, "incorrect": 1}`` for
the baseline bundle while ``src/ml/classifier.py`` defaulted to
``{"incorrect": 0, "correct": 1}`` for transformers. A baseline bundle would
therefore have had its predictions interpreted with the labels swapped at
inference time. Both sides now import the mapping from here.
"""

from typing import Dict

LABELS = ("incorrect", "correct")

LABEL2ID: Dict[str, int] = {"incorrect": 0, "correct": 1}

ID2LABEL: Dict[int, str] = {v: k for k, v in LABEL2ID.items()}

__all__ = ["LABELS", "LABEL2ID", "ID2LABEL"]
