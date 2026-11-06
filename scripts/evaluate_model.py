"""Evaluate a trained model on the held-out test split.

Produces reports/evaluation_report.md and reports/metrics.json.

Usage:
    python scripts/evaluate_model.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.ml.classifier import load_classifier
from src.ml.evaluation import classification_metrics, correlation_metrics, regression_metrics

MODEL_DIR = Path("models/model")
PROCESSED = Path("data/processed")


def main() -> int:
    config_file = MODEL_DIR / "config.json"
    if not MODEL_DIR.exists() or not config_file.exists():
        print("No trained model found at models/model/. Run scripts/train.py first.")
        return 1
    test_file = PROCESSED / "modeling_test.csv"
    if not test_file.exists():
        print("No test split found at data/processed/modeling_test.csv. Run scripts/preprocess_data.py first.")
        return 1

    test = pd.read_csv(test_file)
    classifier = load_classifier(str(MODEL_DIR), mode="production")

    # Stamp provenance first. The demo pipeline writes into the same
    # data/processed/ directory, so without this a synthetic run silently
    # overwrote the report with numbers that read like real measurements.
    # The signal is the explicit sentinel the demo writes -- NOT the `dataset`
    # column, which holds topic ids like "physics_newton" in demo output.
    sentinel = PROCESSED / "SYNTHETIC_DEMO.json"
    datasets = (
        sorted(test["dataset"].dropna().unique().tolist()) if "dataset" in test.columns else []
    )
    is_synthetic = sentinel.exists()
    provenance = {
        "datasets": datasets,
        "test_rows": int(len(test)),
        "synthetic": is_synthetic,
        "synthetic_marker": str(sentinel) if is_synthetic else None,
        "note": (
            "SYNTHETIC DEMO DATA (scripts/demo_pipeline.py). These metrics "
            "measure the demo generator, NOT this system. Do not cite them."
            if is_synthetic
            else "Measured on the held-out split created by preprocessing."
        ),
    }

    report = {
        "model_config": json.loads(config_file.read_text(encoding="utf-8")),
        "provenance": provenance,
    }

    if is_synthetic:
        print(
            "\n" + "!" * 72 + "\n"
            "WARNING: evaluating on SYNTHETIC demo data (datasets=['demo']).\n"
            "Every number below is generated, not measured. It says nothing\n"
            "about model quality. Place a real dataset under data/raw/\n"
            "(see data/DATASETS.md) before citing any figure.\n"
            + "!" * 72 + "\n"
        )

    # --- classification part ---
    labeled = test.dropna(subset=["label"])
    preds = []
    unlabelled = 0
    for _, row in labeled.iterrows():
        result = classifier.predict(row["question"], row["student_answer"])
        if result.label is None:
            # NoOpClassifier (development mode, no trained bundle) returns no
            # label. Coercing that to "incorrect" would manufacture a prediction
            # and produce a meaningless accuracy figure, so it is counted and
            # reported instead.
            unlabelled += 1
            continue
        preds.append(result.label)

    if len(labeled) and unlabelled == len(labeled):
        report["classification"] = {
            "status": "skipped",
            "reason": (
                f"{unlabelled}/{len(labeled)} rows returned no prediction label. "
                "No trained model bundle is loaded (development mode uses "
                "NoOpClassifier), so classification metrics would be fabricated. "
                "Train with: python scripts/train.py --method baseline"
            ),
            "labeled_rows": int(len(labeled)),
            "rows_without_prediction": int(unlabelled),
        }
        print("Classification: SKIPPED - no trained bundle loaded, see metrics.json.")
    elif len(labeled) and unlabelled:
        report["classification"] = {
            "status": "partial",
            "warning": f"{unlabelled} of {len(labeled)} rows had no prediction and were excluded.",
        }
        report["classification"].update(
            classification_metrics(labeled["label"].tolist()[: len(preds)], preds)
        )
        print("Classification (partial):", json.dumps(report["classification"], indent=2))
    elif len(labeled):
        metrics = classification_metrics(labeled["label"].tolist(), preds)
        report["classification"] = metrics
        print("Classification:", json.dumps(metrics, indent=2))
    else:
        report["classification"] = {
            "status": "skipped",
            "reason": "The test split has no labelled rows.",
        }
        print("Classification: SKIPPED - no labelled rows in the test split.")

    # --- similarity-vs-score reference baseline ---
    # This is NOT a metric for the trained classifier. It measures raw TF-IDF
    # cosine similarity against human scores on datasets that carry them
    # (ASAP-AES). Reporting it next to classifier metrics invites the reader to
    # mistake it for model performance, so it is explicitly scoped.
    from src.ml.embeddings import TfIdfEmbeddingProvider

    provider = TfIdfEmbeddingProvider()
    with_scores = test.dropna(subset=["score", "reference_answer", "student_answer"])
    if len(with_scores):
        sims = []
        for _, row in with_scores.iterrows():
            m = provider.similarity_matrix([row["reference_answer"]], [row["student_answer"]])
            sims.append(float(m[0][0]))
        targets = with_scores["score"].astype(float).tolist()
        report["similarity_score_reference"] = {
            "what_this_is": (
                "Raw TF-IDF cosine similarity between the reference and student "
                "answers, compared against human scores. This is a similarity "
                "baseline, NOT the trained model's performance."
            ),
            "n_rows": int(len(with_scores)),
            "regression": regression_metrics(targets, sims),
            "correlation": correlation_metrics(sims, targets),
        }
        print(
            "Similarity-vs-score reference (TF-IDF, not the trained model):",
            json.dumps(report["similarity_score_reference"], indent=2),
        )

    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    (reports_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    md_path = reports_dir / "evaluation_report.md"
    banner = (
        "> **SYNTHETIC DEMO DATA — not a measurement of this system.**\n"
        if is_synthetic
        else ""
    )
    md_path.write_text(
        f"# Model Evaluation Report\n\n"
        f"{banner}"
        f"Model: `{MODEL_DIR}`\n\n"
        f"Datasets evaluated: `{datasets or 'unknown'}`\n\n"
        f"```json\n{json.dumps(report, indent=2)}\n```\n\n"
        "Real benchmark claims require measured results - this file is generated "
        "from the actual evaluation run above.\n",
        encoding="utf-8",
    )
    print(f"\nReport written to {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())