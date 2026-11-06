"""Dataset acquisition helper: prints exact placement instructions and
verifies whatever you have placed under `data/raw/`.

Why this script downloads nothing by default
--------------------------------------------
Every dataset this project targets is distributed behind either a registration
step, a click-through license, or an unstable shared-task page:

* **ASAP-AES** - Kaggle competition data. Requires a Kaggle account and
  acceptance of the competition rules, so it cannot be fetched anonymously.
* **SciEntsBank** - released for the NIST/HUSE AS-2 shared task; mirrors move
  and redistribution terms are per-fileset.
* **BEETLE / BEETLEeval** - released for a SemEval shared task; the host page
  has changed several times.

Rather than point at URLs that rot, silently 404, or quietly return an HTML
error page that later fails as a parse error, this script prints where to put
the files and then checks what you actually placed. A hand-maintained URL that
returns a login page is worse than an honest instruction.

If you add a URL below, it must be a direct, unauthenticated link to the data
file. Use `--verify` afterwards.

Usage:
    python scripts/download_data.py                  # instructions + what is present
    python scripts/download_data.py --dataset beetle # instructions for one dataset
    python scripts/download_data.py --verify         # only check placed files
"""

import argparse
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RAW_DIR = PROJECT_ROOT / "data" / "raw"

# Intentionally empty. See the module docstring: adding a URL here means it is
# a direct unauthenticated download. Kept as a list so adding one is a
# one-line change rather than a code change.
DATASET_URLS: dict[str, list[str]] = {
    "scientbank": [],
    "beetle": [],
    "asap-aes": [],
}

DOWNLOAD_TIMEOUT_SECONDS = 30

INSTRUCTIONS = {
    "scientbank": (
        "SciEntsBank (NIST/HUSE AS-2 shared task; also used for SemEval-2013\n"
        "  Task 7). Obtain it from the NIST LDC / AS-2 task organizers.\n"
        "  Place the JSON filesets under data/raw/scientbank/ (e.g. NAEP.json).\n"
        "  The loader accepts a single file or a directory."
    ),
    "beetle": (
        "BEETLE / BEETLEeval (SemEval-2013 Task 8, short-answer grading).\n"
        "  Obtain it from the SemEval-2013 task organizers or a current\n"
        "  research mirror. Place the XML files under data/raw/beetle/.\n"
        "  The loader accepts a single file or a directory."
    ),
    "asap-aes": (
        "ASAP-AES (Kaggle competition data; full essays, not short answers).\n"
        "  Download from https://www.kaggle.com/c/asap-aes - requires a free\n"
        "  Kaggle account and accepting the competition rules.\n"
        "  Place the CSV/TSV under data/raw/asap/ and the prompts file beside\n"
        "  it. Pass --prompts to scripts/preprocess_data.py to join prompt text."
    ),
}

# Which raw files each loader can actually read.
EXPECTED = {
    "scientbank": {"dir": "scientbank", "patterns": ("*.json",)},
    "beetle": {"dir": "beetle", "patterns": ("*.xml",)},
    "asap-aes": {"dir": "asap", "patterns": ("*.csv", "*.tsv")},
}


def attempt_download(name: str) -> bool:
    """Try any configured URLs. Returns True if at least one file was fetched."""
    out_dir = RAW_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    downloaded = False
    for url in DATASET_URLS.get(name, []):
        filename = Path(url).name
        target = out_dir / filename
        try:
            print(f"Downloading {url} ...")
            with urllib.request.urlopen(  # noqa: S310 - operator-supplied URLs
                url, timeout=DOWNLOAD_TIMEOUT_SECONDS
            ) as response:
                target.write_bytes(response.read())
            print(f"  -> saved to {target}")
            downloaded = True
        except Exception as exc:  # noqa: BLE001 - any failure falls through to instructions
            print(f"  ! failed for {url}: {exc}")
    return downloaded


def verify() -> int:
    """Report, per dataset, whether usable raw files are present."""
    print(f"Checking {RAW_DIR} ...\n")
    total_missing = 0
    for name, spec in EXPECTED.items():
        directory = RAW_DIR / spec["dir"]
        found = []
        if directory.is_dir():
            for pattern in spec["patterns"]:
                found.extend(sorted(p for p in directory.glob(pattern) if p.is_file()))
        found = sorted(set(found))
        if found:
            print(f"[{name}] {len(found)} file(s) in {directory.relative_to(PROJECT_ROOT)}:")
            for path in found[:10]:
                print(f"    {path.name}  ({path.stat().st_size} bytes)")
            if len(found) > 10:
                print(f"    ... and {len(found) - 10} more")
        else:
            total_missing += 1
            print(f"[{name}] MISSING - expected {spec['patterns']} in {directory}")
            for line in INSTRUCTIONS.get(name, "").splitlines():
                print(f"    {line}")
        print()

    if total_missing:
        print(
            f"{total_missing} dataset(s) not present. That is expected: no dataset\n"
            "is bundled with this repository, and none can be downloaded\n"
            "anonymously. Every number reported by this project comes from data\n"
            "you supply, or from the clearly-labelled synthetic demo pipeline."
        )
        return 1
    print("All datasets present. Next: python scripts/preprocess_data.py --all")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Dataset instructions and verification (downloads nothing by default)."
    )
    parser.add_argument(
        "--dataset",
        choices=sorted({*DATASET_URLS, *INSTRUCTIONS}),
        default=None,
        help="Show instructions for one dataset.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Only check which raw files are present under data/raw/.",
    )
    args = parser.parse_args()

    if args.verify:
        return verify()

    if args.dataset:
        if attempt_download(args.dataset):
            print(f"\nDownloaded {args.dataset}. Next: python scripts/preprocess_data.py")
            return 0
        print(INSTRUCTIONS.get(args.dataset, f"No instructions for {args.dataset}."))
        return 0

    print("No automatic download source is configured - see scripts/download_data.py\n")
    for name, message in INSTRUCTIONS.items():
        print(f"[{name}]\n  {message}\n")
    print("Then: python scripts/download_data.py --verify")
    return 0


if __name__ == "__main__":
    sys.exit(main())
