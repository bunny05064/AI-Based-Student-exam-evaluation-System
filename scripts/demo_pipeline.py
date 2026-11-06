"""Demo end-to-end run of the FULL data->model pipeline on synthetic data.

Purpose: prove the pipeline works (validator -> split -> TF-IDF baseline
training -> evaluation) WITHOUT requiring real datasets, torch, or Tesseract.

IMPORTANT:
- The data is SYNTHETIC and clearly labelled "demo".
- The trained model bundle (models/model/) is a PIPELINE-VALIDATION artifact,
  NOT a benchmark and NOT suitable for grading real students.
- Run `scripts/preprocess_data.py` + `scripts/train.py` on real datasets to get
  results that mean anything.

Usage:
    python scripts/demo_pipeline.py [--seed 42] [--n 300]
"""

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))


# ------------------------------------------------------------------------- #
TOPICS = [
    {
        "id": "physics_newton",
        "question": "Explain Newton's Second Law of Motion.",
        "reference": (
            "Force equals mass times acceleration. A larger force on the same "
            "mass produces a greater acceleration."
        ),
        "concepts": ["force", "mass", "acceleration"],
    },
    {
        "id": "physics_kinetic",
        "question": "What is kinetic energy?",
        "reference": (
            "Kinetic energy is the energy an object has because it is moving. "
            "It depends on the mass and the speed of the object."
        ),
        "concepts": ["kinetic energy", "motion", "mass", "speed"],
    },
    {
        "id": "biology_photo",
        "question": "Explain the process of photosynthesis.",
        "reference": (
            "Plants use light energy, water and carbon dioxide to produce "
            "glucose and release oxygen."
        ),
        "concepts": ["light", "water", "carbon dioxide", "glucose", "oxygen"],
    },
    {
        "id": "biology_water",
        "question": "Describe the water cycle.",
        "reference": (
            "Evaporation forms clouds, condensation releases rain, and "
            "precipitation returns water to the ground."
        ),
        "concepts": ["evaporation", "condensation", "precipitation"],
    },
    {
        "id": "chemistry_acid",
        "question": "What is an acid?",
        "reference": (
            "An acid is a substance that releases hydrogen ions and has a pH "
            "below seven."
        ),
        "concepts": ["hydrogen", "ph", "protons"],
    },
    {
        "id": "chemistry_atomic",
        "question": "Define the atomic number of an element.",
        "reference": (
            "The atomic number is the number of protons in the nucleus of an atom."
        ),
        "concepts": ["protons", "nucleus", "atom"],
    },
]

# Several natural phrasings per concept so answers vary (not verbatim gold).
VARIANTS = {
    "force": [
        "Force is the push or pull acting on an object",
        "a force describes the push or pull on a body",
        "force is applied whenever something is pushed or pulled",
    ],
    "mass": [
        "mass is the amount of matter in a body",
        "the mass measures how much matter an object contains",
        "mass tells you how much material is in the object",
    ],
    "acceleration": [
        "acceleration is the rate of change of velocity",
        "acceleration describes how quickly velocity is changing",
        "objects accelerate when their speed or direction changes",
    ],
    "kinetic energy": [
        "kinetic energy is the energy of a moving object",
        "kinetic energy is energy stored in motion",
        "moving objects possess kinetic energy",
    ],
    "motion": [
        "it is the energy linked to the motion of the body",
        "motion of the object gives it this energy",
        "the moving state of the object matters",
    ],
    "speed": [
        "it grows when the speed of the object increases",
        "speed is one important factor",
        "the faster the object moves, the larger it becomes",
    ],
    "light": [
        "light energy from the sun is absorbed by the plant",
        "plants capture light energy",
        "sunlight supplies the needed energy",
    ],
    "water": [
        "plants take up water from the soil",
        "water is absorbed through the roots",
        "roots pull in water from the ground",
    ],
    "carbon dioxide": [
        "carbon dioxide enters through tiny leaf pores",
        "the plant takes carbon dioxide from the air",
        "carbon dioxide is drawn in from the atmosphere",
    ],
    "glucose": [
        "glucose is produced as food for the plant",
        "the process creates glucose sugar",
        "sugar glucose is the final food product",
    ],
    "oxygen": [
        "oxygen is released as a by-product",
        "the plant gives off oxygen gas",
        "oxygen escapes into the air",
    ],
    "evaporation": [
        "water evaporates from oceans and lakes into the air",
        "evaporation turns surface water into vapour",
        "the sun heats water until it evaporates",
    ],
    "condensation": [
        "vapour condenses into clouds high above",
        "condensation forms small droplets in the sky",
        "cooling makes the vapour condense",
    ],
    "precipitation": [
        "precipitation returns water as rain or snow",
        "rain falls to the ground through precipitation",
        "precipitation brings the water back down",
    ],
    "hydrogen": [
        "acids release hydrogen ions when dissolved",
        "hydrogen ions are produced by acids",
        "the acid gives off hydrogen in water",
    ],
    "ph": [
        "acids have a pH smaller than seven",
        "the pH scale places acids below seven",
        "measuring pH shows a value under seven",
    ],
    "protons": [
        "the count of protons decides the element",
        "protons are the positively charged particles",
        "each element has a fixed number of protons",
    ],
    "nucleus": [
        "these protons sit inside the nucleus",
        "the nucleus holds all the protons",
        "protons are packed in the nucleus",
    ],
    "atom": [
        "every atom of the element shares this number",
        "the atomic number belongs to the atom",
        "atoms are identified by this count",
    ],
}

WRONG_POOL = [
    "I do not know.",
    "This topic was not covered in class.",
    "I think the answer is maybe related to energy but I am not sure.",
    "It is hard and there are many factors involved in this question.",
    "We did not learn this yet.",
]


def _concept_sentence(concept: str, rng: random.Random) -> str:
    variants = VARIANTS[concept]
    sentence = rng.choice(variants)
    return sentence[0].upper() + sentence[1:] + "."


def _build_dataset(n_total: int, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    per_topic = max(1, n_total // len(TOPICS))
    for topic in TOPICS:
        concepts = topic["concepts"]
        n = len(concepts)
        for _ in range(per_topic):
            include = rng.choices(range(n + 1), weights=[1, 2, 3, 3, 2, 1][: n + 1])[0]
            chosen = rng.sample(range(n), k=include)
            sentences = [_concept_sentence(concepts[i], rng) for i in chosen]
            if include < n and rng.random() < 0.25:
                sentences.append(rng.choice(WRONG_POOL))
            answer = " ".join(sentences)
            label = "correct" if include >= (n * 2 + 2) // 3 else "incorrect"
            score = round(include / n * 10, 1)
            rows.append(
                {
                    "dataset": topic["id"],
                    "question": topic["question"],
                    "student_answer": answer,
                    "reference_answer": topic["reference"],
                    "label": label,
                    "score": score,
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the demo pipeline on synthetic data.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n", type=int, default=300)
    args = parser.parse_args()

    print(f"[demo] building synthetic dataset (n={args.n}, seed={args.seed})")
    df = _build_dataset(args.n, args.seed)

    from src.data.preprocessing import combine_datasets, preprocess_dataset

    processed = str(BASE / "data" / "processed")
    print("[demo] validating + cleaning + splitting (no leakage: split before features)")
    report = preprocess_dataset(df, "demo", processed, seed=args.seed)
    # The report exposes split_sizes/split_fractions plus an explicit
    # stratification record; there is no single "split" key.
    summary = report.get("summary", {})
    print(f"  rows kept: {summary.get('rows')}")
    print(f"  sizes: {report.get('split_sizes')}")
    print(f"  fractions: {report.get('split_fractions')}")
    print(f"  label distribution: {summary.get('label_distribution')}")
    print(f"  stratified_on: {report.get('stratified_on')}")
    if report.get("stratification_skipped_reason"):
        print(f"  stratification skipped: {report['stratification_skipped_reason']}")

    print("[demo] combining into modeling.csv")
    combine_datasets({"demo": df}, processed, seed=args.seed)

    # Explicit provenance sentinel. evaluate_model.py and the training scripts
    # read this instead of guessing from column values: the demo frame's
    # `dataset` column holds topic ids (physics_newton, ...), not "demo", so
    # inference from the data silently reported synthetic runs as real.
    (Path(processed) / "SYNTHETIC_DEMO.json").write_text(
        json.dumps(
            {
                "synthetic": True,
                "generator": "scripts/demo_pipeline.py",
                "seed": args.seed,
                "n_requested": args.n,
                "warning": (
                    "Metrics computed from this data describe a text generator, "
                    "not this system. Never cite them as model performance."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("[demo] training TF-IDF + logistic regression baseline")
    from src.ml.training import main as run_training

    metrics = run_training(processed, str(BASE / "models" / "model"), method="baseline", seed=args.seed)
    test = metrics.get("test_metrics", {})
    print(f"  test accuracy={test.get('accuracy')} f1={test.get('f1')}")

    print("[demo] running the evaluation report script")
    subprocess.run([sys.executable, str(BASE / "scripts" / "evaluate_model.py")], check=True)

    report_md = BASE / "models" / "model" / "README.md"
    report_md.write_text(
        "## Demo model bundle (PIPELINE-VALIDATION ONLY)\n\n"
        "This artifact was produced by `python scripts/demo_pipeline.py` on a "
        "SYNTHETIC dataset. It exists to prove the data/ML pipeline executes "
        "end-to-end (validator -> leakage-safe split -> TF-IDF + logistic "
        "regression -> evaluation).\n\n"
        "- **Do NOT use it to grade real students.**\n"
        "- It is NOT a benchmark; numbers in reports/metrics.json refer to "
        "synthetic data only.\n"
        "- To get meaningful models: place real datasets under data/raw/ (see "
        "data/DATASETS.md), then run `python scripts/preprocess_data.py --all` "
        "and `python scripts/train.py --method baseline`.\n",
        encoding="utf-8",
    )
    print("[demo] done. model bundle at models/model/ (see README.md inside)")
    return 0


if __name__ == "__main__":
    sys.exit(main())