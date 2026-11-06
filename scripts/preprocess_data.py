"""Reproducible dataset preprocessing.

Usage:
    python scripts/preprocess_data.py --all
    python scripts/preprocess_data.py --dataset scientbank

Loads whatever raw files exist under data/raw/, validates + cleans them,
splits into train/val/test and writes processed CSVs + JSON reports into
data/processed. The split always happens BEFORE any feature engineering, so
no data leakage is possible.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.core.config import BASE_DIR
from src.data.asap_loader import load_asap
from src.data.beetle_loader import load_beetle
from src.data.preprocessing import combine_datasets, preprocess_dataset
from src.data.scientbank_loader import load_scientbank

RAW = BASE_DIR / "data" / "raw"
PROCESSED = BASE_DIR / "data" / "processed"


def _is_prompts_file(path: Path) -> bool:
    """True for an ASAP prompts file (name contains 'prompt').

    The prompts CSV sits *beside* the essays CSV in data/raw/asap/, so without
    this the essay loader could pick the prompts file as the essay file.
    """
    return "prompt" in path.stem.lower()


def load_available(prompts: bool = False, raw_dir: Optional[Path] = None) -> dict:
    raw = raw_dir or RAW
    frames = {}
    scientbank_files = sorted((raw / "scientbank").glob("*.json"))
    if scientbank_files:
        frames["scientbank"] = load_scientbank(str(raw / "scientbank"))

    beetle_files = sorted((raw / "beetle").glob("*.xml"))
    if beetle_files:
        frames["beetle"] = load_beetle(str(raw / "beetle"))

    asap_dir = raw / "asap"
    asap_files = sorted(
        p for p in asap_dir.glob("*") if p.is_file() and p.suffix.lower() in (".csv", ".tsv")
    )
    essay_files = [p for p in asap_files if not _is_prompts_file(p)]
    prompt_files = [p for p in asap_files if _is_prompts_file(p)]
    if essay_files:
        # Only join prompt text when asked (`--prompts`); the loader still marks
        # unresolved prompts as placeholders rather than inventing text.
        prompts_path = str(prompt_files[0]) if (prompts and prompt_files) else None
        frames["asap-aes"] = load_asap(str(essay_files[0]), prompts_path=prompts_path)
    return frames


def main() -> int:
    parser = argparse.ArgumentParser(description="Preprocess educational datasets.")
    parser.add_argument("--all", action="store_true", help="Preprocess every available dataset.")
    parser.add_argument("--dataset", choices=["scientbank", "beetle", "asap-aes"], default=None)
    parser.add_argument(
        "--prompts",
        action="store_true",
        help="Join ASAP prompt text from the prompts file in data/raw/asap/, if present.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    frames = load_available(prompts=args.prompts)
    if not frames:
        print(
            "No raw dataset files found under data/raw/. Run "
            "python scripts/download_data.py and follow the instructions, "
            "then re-run this script."
        )
        return 1

    selected = frames
    if args.dataset and not args.all:
        if args.dataset not in frames:
            print(f"Dataset '{args.dataset}' not placed in data/raw/ yet.")
            return 1
        selected = {args.dataset: frames[args.dataset]}

    reports = {}
    for name, frame in selected.items():
        print(f"Preprocessing {name}: {len(frame)} rows ...")
        reports[name] = preprocess_dataset(
            frame, dataset_name=name, processed_dir=str(PROCESSED), seed=args.seed
        )

    combined_report = combine_datasets(
        {k: v for k, v in selected.items()}, processed_dir=str(PROCESSED), seed=args.seed
    )
    reports["_combined"] = combined_report

    out = PROCESSED / "preprocessing_report.json"
    out.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(f"\nDone. Report at {out}")
    print("Next step: python scripts/train.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())