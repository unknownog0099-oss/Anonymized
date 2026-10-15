# ProvGuard

ProvGuard studies how to reduce false alarms in prompt-injection detection. It
selects legitimate prompts that a detector finds suspicious, checks their intent,
and adds them to the training data. These examples are called **hard negatives**.

The experiments use DistilBERT on PIDS-Bench, with additional prompts from Dolly
and OpenAssistant.

## How it works

1. Score candidate prompts with three baseline detectors.
2. Remove duplicates and prompts too similar to the benchmark data.
3. Keep prompts accepted as benign by both semantic review passes.
4. Retrain the detector and compare it with the baseline, curated examples, and
   randomly selected real prompts.

## Results

Five-seed averages at a detection threshold of 0.5:

| Method | Test F1 (%) | Attack recall (%) | External false-positive rate (%) |
|---|---:|---:|---:|
| Baseline | 98.15 | 98.71 | 20.67 |
| Curated-HN | 98.28 | 98.92 | 18.46 |
| Random-Real | 98.40 | 98.77 | 7.31 |
| ProvGuard-1000 | 98.26 | 98.69 | 4.23 |

The external test contains 208 legitimate prompts. A lower false-positive rate
means fewer legitimate requests are flagged as attacks. ProvGuard reduces false
alarms on this set, but its F1 drops on the separate domain-shift test. Its
advantage over Random-Real is inconclusive on the combined hard-negative test.

See the [full report](reports/FINAL_EXPERIMENT_REPORT.md) for per-seed results,
uncertainty, and limitations, or read the [paper](paper/ProvGuard.pdf).

## Setup

Use Python 3.11 to match the recorded environment. Run these commands from the
repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-reproduction.txt
python -m pip install -e ".[test]"
```

On Linux or macOS, activate the environment with `source .venv/bin/activate`.

## Reproduce

### Check the saved results

```powershell
python scripts/verify_release.py
python -m pytest
```

The verifier checks file hashes and recalculates metrics and statistical
intervals from the saved predictions. This step needs no GPU or model downloads.

### Rerun training

Use a separate copy of the repository: training rewrites result tables and
reports. The training controller requires an NVIDIA GPU with CUDA. The original
runs used an RTX 4060 laptop GPU and PyTorch 2.5.1 with CUDA 12.1.

After setup, install the recorded PyTorch build and start the study:

```powershell
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.cuda.is_available())"
python scripts/download_assets.py
python scripts/run_full_study.py --resume
```

The CUDA check should print `True`. The download step needs internet access.
The study runs six methods across seeds `13, 42, 123, 2024, 7777`: 30 runs in
total, including the 250- and 500-example variants. It uses the included selected
data and review decisions. Trained weights are saved under `outputs/provguard/`.

Check progress in another terminal with `python scripts/status.py`. To continue
after an interruption, rerun the study command with `--resume`.

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for training settings, selection
details, and paper build commands.

## Files

| Path | Contents |
|---|---|
| `src/provguard/` | Selection, training, and evaluation code |
| `scripts/` | Commands for running and checking experiments |
| `data/provguard/` | Candidate prompts, selected examples, scores, and reviews |
| `upstream/` | Benchmark data and baseline code |
| `artifacts/experiments/` | Saved predictions, configurations, and metrics |
| `results_master.csv` | Results for all 30 primary runs |
| `reports/` | Detailed results and statistical comparisons |

Trained weights are not included. Evaluation uses 808 accessible hard-negative
prompts; 664 withheld LMSYS texts are excluded. See the
[benchmark data licenses](upstream/DATA_LICENSES.md) and [license texts](LICENSES/)
for source terms.
