# AI-Based Explainable Student Answer Evaluation & Personalized Feedback System

[![CI](https://github.com/uppu123/AI-Based-Student-exam-evaluation-System/actions/workflows/ci.yml/badge.svg)](https://github.com/uppu123/AI-Based-Student-exam-evaluation-System/actions/workflows/ci.yml)

Automated **short-answer grading** with **explainability** (which concepts were
correct/partial/missing, with quoted evidence) and **personalized learning**
(per-student concept mastery + study recommendations).

Pipeline: `input (text / image / PDF) → [OCR] → concept extraction → rubric
scoring → feedback → recommendations → persistence → API`.

---

## 1. Highlights

- **Explainable by construction** — every score decomposes into per-concept
  statuses (`correct` / `partially_correct` / `missing` / `incorrect`) each
  backed by an evidence sentence from the student answer. No opaque model.
- **Classical baseline out of the box** — runs on TF-IDF + corpus-IDF weighted
  character-shingle coverage (k=4). No GPU, no torch, no embeddings download.
- **Transformer-ready** — swap provider via configuration
  (`EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2`); thresholds are
  provider-aware (see §7).
- **Image & PDF support** — pluggable OCR engines (Tesseract, EasyOCR, mock)
  with pypdf text-layer extraction.
- **Personalized feedback** — exponentially-weighted concept mastery per
  student + deterministic study recommendations (no LLM, always consistent
  with the score).
- **Production-shaped** — Docker/Gunicorn, Postgres, Flask-Migrate migrations,
  BentoML serving bundle, 195-test pytest suite, reproducible dataset pipeline
  (train/val/test split happens **before** any feature engineering).

## 2. Repository layout

```
.
├── app/                    # Flask application
│   ├── core/               # config, exceptions, logging, error handlers
│   ├── models/             # SQLAlchemy models (Student, Evaluation, Concept…)
│   ├── routes/             # REST API blueprints
│   ├── services/           # evaluation orchestration, rubric, feedback, OCR
│   ├── utils/              # text/file/validation helpers
│   └── __init__.py         # application factory
├── src/
│   ├── data/               # dataset loaders, validator, preprocessing
│   ├── ml/                 # embeddings, concept extractor, classifier, training
│   └── ocr/                # OCR engine registry + service
├── templates/  static/     # frontend (vanilla JS)
├── data/                   # raw / interim / processed datasets (git-ignored)
├── models/model/           # trained artifacts (git-ignored, reproducible)
├── tests/                  # pytest suite
├── scripts/                # download_data, preprocess_data, train, evaluate_model
├── bento/                  # BentoML serving service
├── Dockerfile / docker-compose.yml
└── requirements*.txt
```

## 3. Quick start (development, no ML downloads)

**Prereqs:** Python 3.11+.

```bash
pip install -r requirements.txt        # core only (Flask, sklearn, pandas…)
python run.py                          # http://localhost:8000
```

This launches in **development mode**:
- `MODEL_MODE=development` → TF-IDF provider + `NoOpClassifier` (the
  classifier never fakes a prediction — `correctness_prediction` stays `None`).
- `OCR_ENGINE=mock` unless Tesseract is installed (see §6).

Answer-sheet authors are welcome in `data/DATASETS.md`.

## 4. Docker / Gunicorn / Postgres

```bash
docker compose up --build            # app on :8000, Postgres on :5432
```

Deploys `gunicorn --bind 0.0.0.0:8000 run:app` with the existing DB schema
(`db.create_all()` is skipped in production; run migrations for schema
changes). Override secrets via env (`SECRET_KEY`, `DATABASE_URL`).

> Gunicorn is POSIX-only — on Windows use `python run.py`.

## 5. REST API

All endpoints are JSON under `/api/v1`. Errors follow
`{"success": false, "error": {"code", "message", …}}`; internal traces never
leak.

| Method | Path | Description |
| --- | --- | --- |
| GET | `/health` | liveness |
| GET | `/model/info` | loaded model mode/provider/classifier |
| POST | `/evaluate` | text evaluation |
| POST | `/evaluate/image` | image/PDF evaluation (multipart) |
| POST | `/ocr` | OCR a file without grading |
| GET | `/evaluations/<id>` | fetch a stored evaluation |
| POST | `/student` | create a student |
| GET | `/student/<id>/progress` | mastery profile + recent scores |

### `POST /evaluate` — request

```json
{
  "question": "Explain Newton's Second Law of Motion.",
  "answer": "Force depends on the mass of an object and how fast its velocity changes.",
  "max_score": 10,
  "rubric": {
    "name": "Newton's 2nd law",
    "concepts": [
      {"name": "force", "weight": 1, "synonyms": ["f"]},
      {"name": "acceleration", "weight": 1, "synonyms": ["velocity change"]}
    ]
  },
  "student_id": 3,
  "subject": "physics"
}
```

### Response (excerpt)

```json
{
  "success": true,
  "data": {
    "evaluation_id": "a1b2c3d4e5f6a7b8",
    "score": 8.0,
    "max_score": 10,
    "percentage": 80.0,
    "concepts": {
      "correct": ["force", "mass"],
      "partially_correct": ["acceleration"],
      "missing": ["F = ma"],
      "results": [
        {"concept": "force", "status": "correct", "evidence": "Force depends on the mass of an object.", "confidence": 0.98, "weight": 1.0}
      ]
    },
    "feedback": "Overall score: 8/10 (80%). The answer is rated excellent by the rubric engine. ...",
    "improvement_tip": "...",
    "recommendations": [{"topic": "F = ma", "kind": "question", "text": "Attempt 2 conceptual questions..."}],
    "rubric_source": "provided",
    "model_mode": "development"
  }
}
```

Shape only — the numbers above are illustrative, not a measured run.
`confidence` is the measured cosine similarity between the evidence sentence
and the concept, so it is bounded by the embedding provider (TF-IDF and
sentence-transformers produce different ranges for the same answer).

### Rubric resolution order

1. `rubric` in request → `source=provided`
2. curated built-ins (Newton's 2nd Law, Photosynthesis, Water Cycle) → `source=builtin`
3. heuristic keyphrases from the question → `source=auto`

The request's `max_score` is always authoritative.

## 6. OCR

`app/services/ocr_service.py` + `src/ocr/`:

- `OCR_ENGINE=tesseract` (default, requires the Tesseract binary + pytesseract),
- `OCR_ENGINE=easyocr` (heavy, GPU optional — `requirements-ocr.txt`),
- `OCR_ENGINE=mock` (tests/demo — clearly labelled, never production).
- `.pdf` → text-layer extraction via pypdf; scanned/image-only PDFs return a
  clear error suggesting an image upload.

`MAX_UPLOAD_MB`, `OCR_MIN_CONFIDENCE` configure limits/quality gates.

## 7. Concept matching & scoring

`src/ml/concept_extractor.py` + `app/services/rubric_service.py`

1. Each expected concept is a query phrase (name + synonyms + optional
   `incorrect_phrases`) compared against every **sentence** of the answer via
   a `similarity_matrix`; the best sentence becomes the evidence.
2. Status from similarity vs thresholds: `≥ correct → correct`,
   `≥ partial → partially_correct`, a matched `incorrect_phrase` → `incorrect`,
   else `missing`.
3. Score = `Σ weight×factor(status)` over `max_score`, rounded to the nearest
   half-point. `PARTIAL_CREDIT_FACTOR` (0.5), `INCORRECT_PENALTY` (0.0).

Provider-aware defaults (override with `CONCEPT_THRESHOLD_CORRECT` /
`CONCEPT_THRESHOLD_PARTIAL`):

| Provider | correct | partial |
| --- | --- | --- |
| TF-IDF (`tfidf`) | 0.50 | 0.30 |
| sentence-transformers | 0.60 | 0.45 |

Each override is applied independently, so setting only
`CONCEPT_THRESHOLD_CORRECT` takes effect. An override that inverts the
ordering (`correct < partial`) raises rather than silently swapping the two.

## 8. Personalization

`app/services/recommendation_service.py` — exponential moving average mastery
per `(student, concept)`:

```
new_mastery = 0.7·old + 0.3·attempt        # correct=1.0, partial=0.5, else 0.0
```

`strength_of`: `≥0.7 strong`, `<0.5 weak`, else developing. Recommendations
are deterministic templates for missing/incorrect/weak concepts.

## 9. Datasets & training

See `data/DATASETS.md` for details and licensing of SciEntsBank / BEETLE /
ASAP-AES.

```bash
python scripts/download_data.py            # where each dataset comes from (no auto-download)
python scripts/download_data.py --verify   # what is already present under data/raw/
python scripts/preprocess_data.py --all    # validate -> clean -> split -> CSVs
python scripts/train.py --method baseline  # TF-IDF + logistic regression
python scripts/evaluate_model.py           # held-out metrics + reports/
```

The split is seeded and always happens **before** any feature extraction
(prevents leakage). It is stratified only when that is valid — a class with a
single member makes scikit-learn's `stratify=` raise, so preprocessing falls
back to an unstratified seeded split and writes the reason into
`<dataset>_report.json` rather than failing. Near-verbatim overlap between
train and test is measured and reported. Real benchmark numbers belong in
`reports/` (generated by the script, not hand-written); **no dataset is
bundled with this repo** — see `data/DATASETS.md` for provenance, licensing,
and what is not yet verified against the real files.

## 10. BentoML serving

```bash
bentoml serve bento.service:EvaluationService   # dev-mode out of the box
bentoml build -f bentofile.yaml
bentoml containerize ai-answer-evaluation:latest
```

Endpoints: `health`, `metadata` (provider / thresholds / rubric config),
`evaluate_text` (same scoring logic as `/api/v1/evaluate`, stateless, no DB).
Leave `MODEL_MODE=development` until a trained bundle exists in `models/`.

## 11. Configuration reference

`.env` (see `.env.example`):

| Variable | Default | Meaning |
| --- | --- | --- |
| `FLASK_ENV` | `development` | env label; `production` skips `create_all` |
| `SECRET_KEY` | dev value | Flask secret — change in production |
| `DATABASE_URL` | `sqlite:///app.db` | SQLAlchemy URL |
| `MODEL_MODE` | `development` | `development` → TF-IDF + noop classifier; `production` → load trained bundle |
| `MODEL_PRELOAD` | `false` | load model at startup |
| `EMBEDDING_MODEL` / `EMBEDDING_FALLBACK` | SBERT / `tfidf` | provider + fallback |
| `OCR_ENGINE` | `tesseract` | `tesseract` \| `easyocr` \| `mock` |
| `MAX_UPLOAD_MB`, `MAX_ANSWER_CHARS`, `MAX_SCORE_ALLOWED` | 10 / 10000 / 100 | limits |
| `CONCEPT_THRESHOLD_*` | provider defaults | matching thresholds |
| `PARTIAL_CREDIT_FACTOR`, `INCORRECT_PENALTY` | 0.5 / 0.0 | rubric scoring |
| `CORS_ORIGINS` | `*` | comma-separated allowed origins |

## 12. Testing & quality gates

```bash
python -m pytest                          # unit + integration (deselect -m "not integration")
python -m pytest --cov=app --cov=src --cov=scripts --cov=bento
python scripts/demo_pipeline.py           # end-to-end data -> model smoke test (synthetic)
```

Test doubles (`tests/mocks.py`) are loudly fake (word-coverage similarity,
`NoOpClassifier`, mock OCR) so suites are deterministic and never need
torch/Tesseract/dataset downloads. Nothing in the suite reaches the network:
the Hugging Face tokenizer tests stub `transformers` rather than calling
`from_pretrained`, so the full run takes a few seconds.

GitHub Actions CI (`.github/workflows/ci.yml`) runs pyflakes, a migration
drift check, pytest **with a 65% coverage gate**, and the demo-pipeline smoke
test on Python 3.11 and 3.12 (the badge above links to the latest run).
Measured coverage locally is 72% across `app`/`src`/`scripts`/`bento`; the
gate is set below that on purpose, since a threshold that only passes on one
machine is not a quality gate.

## 13. Roadmap

- Knowledge-tracing student models (e.g. DKT) on the persisted attempt
  history (schema already supports it).
- Multi-head rubric weights & objective-answer type detection.
- Dockerized training job + model registry.