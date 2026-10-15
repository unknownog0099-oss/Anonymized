# Reproducibility

## Inspect the completed evidence

`python scripts/verify_release.py` independently recalculates seven primary
metrics from all 30 archived prediction sets and verifies all six aggregate
paired seed/shared-prompt bootstrap intervals. It also verifies the release
manifest and frozen scientific bytes. Model-weight verification requires the
excluded original weights and is not claimed by this distribution.

Archived evidence lives in `artifacts/experiments/`. Runtime training writes to
`outputs/provguard/`; historical registries under `artifacts/` are separate from
fresh controller registries under `reports/`. The recorded per-run configurations
are authoritative for training dependencies; `artifacts/software_versions.json`
records the release environment and `requirements-reproduction.txt` pins its
non-Torch dependencies. Install a CUDA-enabled PyTorch matching your hardware
before those pins when reproducing training. The original study used Torch
2.5.1+cu121 on an RTX 4060 laptop GPU. CPU-only training is rejected by the study.

All arms use DistilBERT-base-uncased, length 256, two epochs, effective batch 16,
learning rate 5e-5, AdamW weight decay 0.01, maximum gradient norm 1.0, and original
validation-F1 checkpoint selection. Every evaluation uses threshold 0.5. The
predefined primary K is 1000; K=250 and K=500 are nested ablations.

The upstream revision is 7984e2d6d95e98e914d2b0cd92574b1752614b10. Vendored code,
public benchmark files, freeze manifest, and original third-party notices are
included. Restricted text remains withheld; reconstruction is not required.

## Repeat training

Use a separate checkout/copy so the released evidence stays unchanged. With
CUDA PyTorch and the pinned dependencies installed:

```powershell
python scripts/download_assets.py
python scripts/finish_study.py --resume
python scripts/status.py
```

The first command downloads pretrained DistilBERT assets. The controller trains
and evaluates all 30 required runs using the provided frozen augmentation and
the original validation split. New model/tokenizer assets and probabilities are
saved under `outputs/`; no pretrained or trained weights are included in Git.
No training was performed during release packaging.

## Selection and adjudication

Scores from baseline seeds 13/42/123 are retained in `scored_candidates.csv`;
the historical exact-text cache supports the recorded selector reuse. Dolly
uses instruction fields only; OASST1 uses English nondeleted root prompters.
The persisted TF-IDF char_wb 3-5 representation was fitted once on the available
reference/candidate union. Exact and cosine >=0.90 matches are excluded. Its
original representation, sparse reference matrix, and hashes are provided.

The two review passes, complete full-text inputs, prompts, labels, and reasons
are under `data/provguard/semantic_batches/`. Both BENIGN accepts, either
INJECTION rejects, and remaining disagreement excludes. These are automated
model judgments, not human annotations. Shared errors and an unexposed exact
serving-model identifier limit reproducibility of new reviewer decisions.

Do not refit the exclusion index to another population or silently regenerate
semantic decisions. `python scripts/provguard.py build` rechecks the recorded
selection, matching quotas, and lexical exclusion from the included evidence;
it writes regenerated augmentation files, so run it in a separate checkout.

## Paper

With standard IEEEtran/TikZ/Helvetica and the other declared TeX packages:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error ProvGuard.tex
pdflatex -interaction=nonstopmode -halt-on-error ProvGuard.tex
```

No BibTeX or project asset files are required by the root single-file source.
Alternatively install Tectonic on PATH and run `python scripts/build_paper.py`.
Editable figure/table/macros generation uses `scripts/build_paper_assets.py`.
`scripts/make_single_tex.py` converts the preserved modular manuscript plus
generated assets back into the single-file snapshot. It overwrites the root
source. The release PDF has identifying/time metadata removed; its text and
six-page layout are preserved.

## Anonymity and integrity

Release packaging changes local path/process/time metadata and venue-specific
labels. Numerical metric strings, dataset text, labels, probabilities, frozen
benchmark files, augmentation sets, semantic decisions, and the persisted
lexical index are preserved. Historical hashes identify the original experiment
artifacts; the release manifest separately binds the exported files after
metadata sanitization. Publication figure metadata was stripped, while geometry
and text are preserved. The original working experiments were kept intact.

No project-author name, affiliation, personal remote, email, or Git history is
included. Standard licenses and literature/source authors remain credited.
