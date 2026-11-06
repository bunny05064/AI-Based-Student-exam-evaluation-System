## Demo model bundle (PIPELINE-VALIDATION ONLY)

This artifact was produced by `python scripts/demo_pipeline.py` on a SYNTHETIC dataset. It exists to prove the data/ML pipeline executes end-to-end (validator -> leakage-safe split -> TF-IDF + logistic regression -> evaluation).

- **Do NOT use it to grade real students.**
- It is NOT a benchmark; numbers in reports/metrics.json refer to synthetic data only.
- To get meaningful models: place real datasets under data/raw/ (see data/DATASETS.md), then run `python scripts/preprocess_data.py --all` and `python scripts/train.py --method baseline`.
