# Format fixtures

**These files are not dataset data.**

Every file here is hand-written to exercise one parser's schema handling. The
texts are generic placeholder sentences written for this repository; they are
not samples of ASAP-AES, BEETLEeval, or SciEntsBank and must not be used to
train, evaluate, or report any result.

They exist so `tests/test_loaders.py` can prove each loader produces the
standardized frame for the documented shape, and fails loudly when a field is
renamed or dropped.

## What is verified and what is not

| Loader | Fixture shape | Verified against real data? |
| --- | --- | --- |
| `asap_loader.py` | CSV with `essay_set`, `essay`, `domain1_score`; prompts CSV keyed by `PromptID` | **No** — column names come from the public ASAP description, not from a downloaded copy |
| `beetle_loader.py` | Attribute-per-answer XML *and* element-per-answer XML | **No** — both shapes are supported defensively; which one the released files use has not been confirmed |
| `scientbank_loader.py` | JSON list of records with `correct_answer`, `student_answer`, `SA` | **No** — key names come from the public AS-2 task description |

Before using a loader on real data, open the real file and confirm the field
names. If they differ, add the name to the loader's constants rather than
renaming columns in the data. `data/DATASETS.md` records the same caveat.

## Files

- `asap_essays.csv`, `asap_prompts.csv`
- `beetle_answers_attribute.xml`, `beetle_answers_element.xml`
- `scientbank_records.json`

## Running the loader tests

```bash
python -m pytest tests/test_loaders.py -v
```
