"""Reproducible training pipeline.

Two supported methods:

1. ``baseline`` (default): TF-IDF + logistic regression - runs anywhere with
   scikit-learn, no GPU needed. This is Experiment 1/2's classical baseline.
2. ``transformer``: fine-tune a Hugging Face encoder (requires requirements-ml.txt
   and real hardware/keyed config); guarded so a missing torch raises a clear error.

Artifacts saved to ``models/model/``:
    config.json, model.joblib (baseline) or weights/, tokenizer/, metrics.json
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from app.core.exceptions import ModelLoadError
from src.ml.evaluation import classification_metrics
from src.ml.labels import LABEL2ID


@dataclass
class TrainingConfig:
    method: str = "baseline"  # "baseline" | "transformer"
    seed: int = 42
    val_frac: float = 0.15
    test_frac: float = 0.15
    tokenizer_path: str = "sentence-transformers/all-MiniLM-L6-v2"
    transformer_model: str = "microsoft/deberta-v3-xsmall"
    epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 2e-5
    max_length: int = 256
    output_dir: str = "models/model"


def text_features(df: pd.DataFrame) -> pd.Series:
    """Combine question & answer into a single feature string."""
    q = df["question"].astype(str).fillna("")
    a_col = "answer" if "answer" in df.columns else "student_answer"
    a = df[a_col].astype(str).fillna("")
    return (q + " [SEP] " + a).str.strip()


def _usable_strata(df: pd.DataFrame, label_col: str) -> bool:
    """True when sklearn can stratify on ``label_col``.

    sklearn raises if any class has a single member, which is common for small
    hand-written answer keys, so fall back to an unstratified split instead of
    failing the whole training run.
    """
    if label_col not in df.columns:
        return False
    values = df[label_col]
    if not values.notna().all() or values.nunique() < 2:
        return False
    return int(values.value_counts().min()) >= 2


def stratified_split(
    df: pd.DataFrame,
    label_col: str = "label",
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified train/validation/test split - done BEFORE any feature work
    to prevent data leakage (no overlap, no test-time tuning).
    """
    from sklearn.model_selection import train_test_split

    if val_frac + test_frac >= 1.0:
        raise ModelLoadError("val_frac + test_frac must be < 1.0.")

    stratify = df[label_col] if _usable_strata(df, label_col) else None
    train, temp = train_test_split(
        df, test_size=val_frac + test_frac, stratify=stratify, random_state=seed
    )
    val, test = train_test_split(
        temp,
        test_size=test_frac / (val_frac + test_frac),
        stratify=temp[label_col] if stratify is not None else None,
        random_state=seed + 1,
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def build_tfidf_pipeline(seed: int = 42):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(ngram_range=(1, 2), max_features=20000, sublinear_tf=True),
            ),
            ("clf", LogisticRegression(max_iter=2000, random_state=seed)),
        ]
    )


def train_baseline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seed: int = 42,
) -> Tuple[object, dict]:
    pipeline = build_tfidf_pipeline(seed)
    pipeline.fit(text_features(train_df), train_df["label"])

    val_pred = list(pipeline.predict(text_features(val_df)))
    # Probability of the PREDICTED class, not of the "correct" class - named
    # accordingly so it is not mistaken for correctness confidence.
    val_pred_prob = [float(np.max(p)) for p in pipeline.predict_proba(text_features(val_df))]
    test_pred = list(pipeline.predict(text_features(test_df)))

    metrics = {
        "method": "baseline",
        "val_metrics": classification_metrics(val_df["label"].tolist(), val_pred),
        "test_metrics": classification_metrics(test_df["label"].tolist(), test_pred),
        "val_predicted_class_probability": {
            "mean": round(float(np.mean(val_pred_prob)), 4),
            "median": round(float(np.median(val_pred_prob)), 4),
        },
    }
    return pipeline, metrics


def train_transformer(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: TrainingConfig,
) -> dict:
    """Fine-tune a Hugging Face sequence classifier. Requires torch."""
    try:
        import torch  # noqa: F401, PLC0415
        from torch.utils.data import DataLoader, Dataset  # noqa: PLC0415
        from transformers import (
            AdamW,
            AutoModelForSequenceClassification,
            AutoTokenizer,
            get_linear_schedule_with_warmup,
        )
    except ImportError as exc:  # pragma: no cover - guard for environments w/o torch
        raise ModelLoadError(
            "torch/transformers are required for transformer training. "
            "Install requirements-ml.txt or use --method baseline."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)
    label2id = dict(LABEL2ID)

    class QADataset(Dataset):
        def __init__(self, df: pd.DataFrame):
            self.enc = tokenizer(
                df["question"].tolist(),
                df["answer"].tolist(),
                padding=True,
                truncation=True,
                max_length=config.max_length,
                return_tensors="pt",
            )
            self.labels = torch.tensor(df["label"].map(label2id).tolist())

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return {
                k: v[idx] for k, v in self.enc.items()
            } | {"labels": self.labels[idx]}

    train_ds = QADataset(train_df)
    val_ds = QADataset(val_df)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size)

    model = AutoModelForSequenceClassification.from_pretrained(
        config.transformer_model, num_labels=2
    )
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    total_steps = len(train_loader) * config.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, total_steps)

    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(config.epochs):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            scheduler.step()

        model.eval()
        preds, truths = [], []
        with torch.inference_mode():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(**batch).logits
                preds.extend(logits.argmax(-1).cpu().tolist())
                truths.extend(batch["labels"].cpu().tolist())
        val_pred_labels = [("correct" if p == 1 else "incorrect") for p in preds]
        val_true_labels = [("correct" if t == 1 else "incorrect") for t in truths]
        print(f"epoch {epoch+1}: {classification_metrics(val_true_labels, val_pred_labels)}")

    id2label = {v: k for k, v in label2id.items()}
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    # Read the HF config that save_pretrained just wrote, add the label map,
    # then write it back. The read must happen BEFORE opening the file for
    # writing: "w" truncates immediately, so reading afterwards would always
    # raise JSONDecodeError on an empty string.
    config_path = Path(config.output_dir) / "config.json"
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    cfg["label2id"] = label2id
    cfg["id2label"] = id2label
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    return {"method": "transformer", "epochs": config.epochs, "device": str(device)}


def save_model_bundle(
    artifacts_dir: str,
    method: str,
    label2id: Dict[str, int],
    metrics: dict,
    tokenizer_name: str,
) -> Path:
    """Write config.json + metrics.json describing a saved model bundle.

    For the transformer method ``save_pretrained`` has already written a full
    Hugging Face ``config.json``; it is merged into rather than replaced, since
    overwriting it with this summary would strip the architecture fields that
    ``AutoConfig.from_pretrained`` needs at load time.

    The summary deliberately uses ``bundle_type`` and never ``model_type``:
    Hugging Face's ``model_type`` (e.g. "deberta-v2") selects the config class
    and must be preserved untouched.
    """
    out = Path(artifacts_dir)
    out.mkdir(parents=True, exist_ok=True)
    config_path = out / "config.json"

    summary = {
        "bundle_type": "transformer" if method == "transformer" else "baseline",
        "label2id": label2id,
        "id2label": {str(v): k for k, v in label2id.items()},
        "tokenizer": tokenizer_name,
        "framework": "pytorch" if method == "transformer" else "sklearn",
        "seed_note": "No trained weights are committed to the repository.",
    }

    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        existing.update(summary)
        config = existing
    else:
        config = summary

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return config_path


def load_processed_dataset(processed_dir: str) -> pd.DataFrame:
    path = Path(processed_dir) / "modeling.csv"
    if not path.exists():
        raise ModelLoadError(
            f"No processed dataset found at {path}. Run scripts/preprocess_data.py first."
        )
    return pd.read_csv(path)


def load_pre_splits(processed_dir: str) -> Dict[str, pd.DataFrame]:
    """Load the train/val/test CSVs written by ``preprocess_dataset``.

    Training must reuse the split that preprocessing already made. Re-splitting
    here with a fresh call would move rows across the boundary and invalidate the
    "test rows never influence training" guarantee recorded in the report.
    Returns an empty dict when the split files are absent.
    """
    base = Path(processed_dir)
    splits: Dict[str, pd.DataFrame] = {}
    for name in ("train", "val", "test"):
        path = base / f"modeling_{name}.csv"
        if not path.exists():
            return {}
        splits[name] = pd.read_csv(path)
    return splits


def main(
    data_path: str,
    out_path: str,
    method: str = "baseline",
    seed: int = 42,
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_length: int = 256,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> dict:
    pre_splits = load_pre_splits(data_path)
    if pre_splits:
        train, val, test = (
            pre_splits["train"],
            pre_splits["val"],
            pre_splits["test"],
        )
        split_source = "pre-split CSVs from preprocessing (no re-splitting)"
    else:
        df = load_processed_dataset(data_path)
        train, val, test = stratified_split(
            df, seed=seed, val_frac=val_frac, test_frac=test_frac
        )
        split_source = f"split here (seed={seed}, stratified where possible)"

    if method == "baseline":
        pipeline, metrics = train_baseline(train, val, test, seed=seed)
        import joblib

        model_dir = Path(out_path)
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, model_dir / "model.joblib")
    else:
        metrics = train_transformer(
            train,
            val,
            test,
            TrainingConfig(
                method="transformer",
                output_dir=out_path,
                seed=seed,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                max_length=max_length,
            ),
        )

    metrics["split_source"] = split_source
    metrics["n_rows"] = {"train": len(train), "val": len(val), "test": len(test)}
    save_model_bundle(
        out_path,
        method,
        dict(LABEL2ID),
        metrics,
        tokenizer_name="sentence-transformers/all-MiniLM-L6-v2",
    )
    return metrics