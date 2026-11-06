"""Tokenizers: Hugging Face AutoTokenizer when available, else a simple fallback.

The fallback tokenizer exists so the project imports and runs without the HF
stack; transformer training uses the real tokenizer.
"""

import hashlib
import re
from typing import List, Optional

from app.core.exceptions import ModelLoadError

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[._+\-=^/*][A-Za-z0-9]+)*|[\w]+|[^\w\s]")


class SimpleTokenizer:
    """Deterministic whitespace-free fallback tokenizer for CNNs/baselines."""

    name = "simple"

    def __init__(self, vocab_size: int = 32_000) -> None:
        self.vocab_size = vocab_size

    def tokenize(self, text: str) -> List[str]:
        return [t for t in _TOKEN_RE.findall(text) if t]

    def encode(self, text: str) -> List[int]:
        return [self._hash(tok) for tok in self.tokenize(text)]

    def _hash(self, token: str) -> int:
        digest = hashlib.md5(token.encode("utf-8", errors="ignore")).digest()
        return int.from_bytes(digest[:4], "big") % (self.vocab_size - 1) + 1

    def get_vocab_size(self) -> int:
        return self.vocab_size


class HFTokenizer:
    """Wrap a Hugging Face tokenizer (lazy import of transformers)."""

    def __init__(self, path_or_name: str) -> None:
        self.path_or_name = path_or_name
        self._tokenizer = None

    def _load(self):
        if self._tokenizer is not None:
            return
        try:
            from transformers import AutoTokenizer  # noqa: PLC0415
        except ImportError as exc:
            raise ModelLoadError(
                "transformers not installed; install requirements-ml.txt."
            ) from exc
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.path_or_name)
        except Exception as exc:  # noqa: BLE001
            raise ModelLoadError(
                f"Failed to load tokenizer '{self.path_or_name}'."
            ) from exc

    def tokenize(self, text: str) -> List[str]:
        self._load()
        return self._tokenizer.tokenize(text)

    def encode(self, text: str) -> List[int]:
        self._load()
        return self._tokenizer.encode(text, add_special_tokens=True)

    def get_vocab_size(self) -> int:
        self._load()
        return self._tokenizer.vocab_size

    @property
    def name(self) -> str:
        return self.path_or_name


def load_tokenizer(path_or_name: Optional[str] = None) -> object:
    """Load an HF tokenizer; fall back to the simple tokenizer."""
    if not path_or_name or path_or_name == "simple":
        return SimpleTokenizer()
    return HFTokenizer(path_or_name)