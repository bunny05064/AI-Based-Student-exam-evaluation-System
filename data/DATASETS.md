# Datasets & the data pipeline

## The short version

**No dataset ships with this repository.** All three target datasets require
registration, a click-through license, or a shared-task page whose location
moves. `data/raw/` is git-ignored and starts empty.

Every number this project reports therefore comes from either:

1. data **you** place under `data/raw/`, or
2. the clearly-labelled **synthetic demo pipeline** (`scripts/demo_pipeline.py`),
   which exists to prove the code runs end to end and is not a benchmark.

There are no hard-coded accuracy figures anywhere in the code or docs, and no
metric is reported that was not computed from a real run.

```bash
python scripts/download_data.py            # where to put each dataset
python scripts/download_data.py --verify    # what is present right now
```

## Supported datasets

| Dataset | Task | Shape | Usable for | Loader |
| --- | --- | --- | --- | --- |
| SciEntsBank | short-answer correctness (science) | JSON records | classification, concept work | `src/data/scientbank_loader.py` |
| BEETLE / BEETLEeval | short-answer correctness (tutorial dialogue) | XML records | classification | `src/data/beetle_loader.py` |
| ASAP-AES | automated **essay** scoring | CSV/TSV + prompts | regression only | `src/data/asap_loader.py` |

Provenance and access notes are in the next section. Read them before
distributing anything derived from these datasets — the licenses differ and
at least one is per-fileset.

### Access & licensing

- **SciEntsBank** — released for the NIST/HUSE Automated Short-Answer
  Scoring (**AS-2**) shared task, and used for SemEval-2013 Task 7. Obtain it
  through the NIST LDC or the AS-2 organizers. Licensing is **per fileset**;
  check the terms of the specific fileset you use before redistributing
  derived artifacts or trained models.
- **BEETLE / BEETLEeval** — released for **SemEval-2013 Task 8** (grading of
  student responses in short-answer educational text). The task host page has
  moved more than once, so the download link is not hard-coded here; find the
  current location from the SemEval-2013 task description or a research mirror.
- **ASAP-AES** — Kaggle competition data, `https://www.kaggle.com/c/asap-aes`.
  Requires a free Kaggle account **and** accepting the competition rules.
  Research use; not redistributable.

`scripts/download_data.py` deliberately has **no** hard-coded download URLs.
A stale URL that returns a login page fails later as a confusing parse error,
which is worse than an explicit instruction. The script prints where to put
files and verifies what you placed.

## The canonical frame

Every loader normalizes into one shape:

```
dataset | question | student_answer | reference_answer | label | score | source_file
```

- `label` — `"correct"` / `"incorrect"`, or `None` when not derivable. The
  space is **binary**: partially-correct responses collapse to `incorrect`
  (`src/data/dataset_loader.py`). That is a deliberate loss of information;
  the underlying labels are not recoverable after loading.
- `score` — optional human score. Only ASAP-AES populates it.
- `source_file` — the originating file, so any row can be traced back.

A dataset with no labels cannot train the classifier. `label = None` rows are
kept rather than dropped, and are excluded from classification metrics.

### ASAP-AES is regression-only

ASAP ships human *scores* on 1–6 essay sets, not correct/incorrect labels, so
`label` is always `None`. Two further caveats:

- `domain1_score` is a sum over two raters and is **not comparable across
  essay sets**. Do not pool sets and treat the number as one scale.
- If the prompts file is absent or its join key does not match, every question
  becomes an explicit `[prompt text unavailable for essay set N]` placeholder
  and `frame.attrs["prompt_placeholders"]` reports the count. The loader does
  not substitute invented prompt text.

## Are the loaders verified against the real data?

**No.** This is the most important caveat in this file.

Each loader's field names come from the public task descriptions, and each is
tested against a hand-written **format fixture** in `data/fixtures/` that
mirrors the documented shape. No real dataset file has been run through them
in this repository.

Before relying on a loader:

1. Open the real file and list its top-level keys or XML tags.
2. Compare against the constants at the top of the loader
   (`QUESTION_KEYS`, `ANSWER_TAGS`, `PROMPT_COLUMNS`, …).
3. If a name differs, **add it to the loader's constants**. Do not rename
   columns in your copy of the data — that breaks reproducibility against the
   original release.

A schema mismatch is reported, not hidden: each loader raises `DatasetError`
listing the keys or tags it actually found, rather than returning an empty
frame that looks like an empty download.

### Fixtures are not data

`data/fixtures/` contains placeholder text written for this repository to
exercise the parsers. It is not a sample of any dataset and must never be used
to train, evaluate, or report a result. See `data/fixtures/README.md`.

Two loader bugs were found and fixed by writing these fixtures; both are
covered by regression tests in `tests/test_loaders.py`:

- SciEntsBank's `correct_answer` was being read as a labelled *correct student
  answer*. It is the gold answer. That leaked the answer key into the training
  data and would have inflated every reported metric.
- BEETLE releases records with the answer in an **attribute**
  (`<Answer answer="..." label="..."/>`) as well as in a child element. The
  element-only parser produced zero rows from attribute-style files.

## Preprocessing

```bash
python scripts/preprocess_data.py --all              # or --dataset <name>
python scripts/preprocess_data.py --all --prompts    # join ASAP prompt text
```

`src/data/preprocessing.py`:

1. **validate** — required columns, empties, duplicates, label values.
2. **normalize** — strip/NFKC text, drop empty rows, dedupe exact
   (question, answer), coerce numeric score, map labels to the binary space.
3. **clean** — Unicode NFKC, whitespace collapse. Math tokens like `F=ma` are
   preserved and stopwords are **not** aggressively removed.
4. **split** — train/val/test on **rows only**, before any TF-IDF fit or
   embedding construction, so no test row can influence feature extraction.

Stratification is applied only when it is valid, and the report says what
actually happened:

```json
"stratified_on": null,
"stratification_skipped_reason": "Column 'label' has classes with fewer than 2 members (smallest class = 1); sklearn cannot stratify it."
```

A class with a single member makes `scikit-learn`'s `stratify=` raise. Rather
than crash, preprocessing falls back to an unstratified seeded split and
records the reason. Small hand-written answer keys hit this routinely.

Outputs go to `data/processed/`:

```
<dataset>_{train,val,test,all}.csv   + <dataset>_report.json
modeling.csv + modeling_{train,val,test}.csv   (after --combine)
```

## Training

```bash
python scripts/train.py --method baseline      # TF-IDF + logistic regression
python scripts/train.py --method transformer   # needs requirements-ml.txt
python scripts/evaluate_model.py
```

Training **reuses the split preprocessing already made**
(`modeling_{train,val,test}.csv`). It does not re-split: a fresh split would
move rows across the boundary and invalidate the guarantee recorded in the
report. It falls back to splitting only when those files are absent, and
records which path it took in `metrics.json["split_source"]`.

`--method baseline` is the default and runs anywhere scikit-learn runs, with no
GPU and no model download. Artifacts land in `models/model/`
(`config.json`, `model.joblib`, `metrics.json`). **No trained weights are
committed**; reproduce them with the scripts above.

`label2id` is defined once in `src/ml/labels.py` and shared by training and
inference. It previously differed between the two — the baseline bundle wrote
`{"correct": 0, "incorrect": 1}` while the classifier read
`{"incorrect": 0, "correct": 1}` — which would have swapped every prediction
from a baseline model.

## Reading the numbers honestly

- `reports/metrics.json` and `reports/evaluation_report.md` are **generated**.
  If they are missing, no run has happened; do not quote figures from the docs.
- The synthetic demo in `scripts/demo_pipeline.py` exists to show the pipeline
  runs. Its metrics are computed on generated data and mean nothing about
  model quality.
- Improvements are only credible against the held-out split that never touched
  training. The preprocessing report also records label distribution and
  duplicate counts so near-duplicate leakage across splits is visible.
