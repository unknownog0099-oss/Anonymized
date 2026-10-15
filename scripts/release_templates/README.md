# ProvGuard

Anonymous research code and measured artifacts for **Confusion-Selected Hard
Negatives for Reducing Prompt-Injection Over-Defense under Distribution Shift**.

ProvGuard ranks naturally occurring Dolly/OpenAssistant prompts using three
baseline DistilBERT detectors, accepts candidates only after two automated
semantic-intent adjudication passes, and retrains the same classifier with
verified-benign augmentation. This is a data-selection study.

## Measured study

All 30 required runs completed: six arms with seeds 13, 42, 123, 2024, and 7777.
The fixed threshold is 0.5 and primary augmentation size is 1000.

| Method | IID F1 (%) | Attack recall (%) | External HB FPR (%) |
|---|---:|---:|---:|
| Baseline | 98.15 | 98.71 | 20.67 |
| Curated-HN | 98.28 | 98.92 | 18.46 |
| Random-Real | 98.40 | 98.77 | 7.31 |
| ProvGuard-1000 | 98.26 | 98.69 | 4.23 |

Values are five-seed means. External coverage is 208 prompts; total accessible
hard-benign coverage is 808 (600 curated + 208 external). The 664 withheld LMSYS
texts are excluded. ProvGuard improves over Random-Real on the external endpoint;
the aggregate comparison is inconclusive. Curated-HN wins the curated-heavy
aggregate and ProvGuard reduces domain-OOD F1. Full results, sample SD, uncertainty,
and limitations are in [the report](reports/FINAL_EXPERIMENT_REPORT.md) and paper.

## What is included

- `src/`, `scripts/`, `tests/`, `configs/`: pipeline, controls, and checks.
- `upstream/`: attributed baseline/building code and eight frozen public files.
- `data/provguard/`: exact selected/candidate data, cached selector scores,
  paired semantic reviews, and the original persisted lexical exclusion index.
- `results_master.csv`, `reports/`: measured results and methodological audits.
- `artifacts/experiments/`: per-run configurations, histories, metrics, and all
  seven probability exports for 30 primary runs and six completed source checks.
- `ProvGuard.tex`, `paper/ProvGuard.pdf`, `figures/paper/`: anonymous paper and
  editable publication graphics. The root TeX file is self-contained.
- `artifacts/release_manifest.json`: checksums for this sanitized release.

Model weights, optimizer checkpoints, caches, runtime locks, personal paths,
private environment files, and Git history are excluded. Completed prediction
evidence is archived separately from fresh training outputs. The optional
DeBERTa attempt failed technically; further random-sample reviews are incomplete.
Neither is presented as completed evidence. Semantic decisions are automated,
not human annotation; human-review queue fields remain empty.

## Install and verify

Python 3.10+ is required; the recorded environment used Python 3.11.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[test,paper]"
.\.venv\Scripts\python.exe scripts\verify_release.py
.\.venv\Scripts\python.exe -m pytest
```

The release verifier checks file hashes, frozen benchmark/augmentation bindings,
prediction-derived metrics, paired statistical intervals, and anonymity indicators
without model downloads or training. [REPRODUCIBILITY.md](REPRODUCIBILITY.md)
explains exact software versions, retraining, selection, and paper compilation.

## Upload

Upload the contents of this folder as the repository root, using the Git commands
in [UPLOAD_TO_GITHUB.md](UPLOAD_TO_GITHUB.md). Some scientific artifacts exceed
the browser upload limit; the Git CLI supports all included file sizes.
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for preserved source credits
and data-license links. Dataset text and cited authors are not project-author
identity, and their attribution has not been removed.
