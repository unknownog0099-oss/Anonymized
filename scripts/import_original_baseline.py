"""Register the faithfully reproduced seed-42 model without retraining it."""
import importlib.metadata
import json
import shutil
import sys
from pathlib import Path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))
from provguard.core import BENCH, OUTPUT, integrity, save_json, sha

integrity()
original = root / "outputs/original_baseline"
if json.loads((original / "status.json").read_text())["returncode"] != 0:
    raise RuntimeError("Original baseline reproduction did not complete.")
source_summary = json.loads((root / "upstream/outputs/distilbert_baseline/summary.json").read_text())
out = OUTPUT / "baseline/seed_42"
if out.exists():
    raise FileExistsError(out)
out.mkdir(parents=True)
shutil.copytree(root / "upstream/models/distilbert", out / "model")
shutil.copy2(original / "evaluation.json", out / "evaluation.json")
for file in original.glob("*_predictions.csv"):
    shutil.copy2(file, out / file.name)
save_json(out / "config.json", {"method": "baseline", "seed": 42, "model": "distilbert-base-uncased", "training_rows": 27093, "augmentation_size": 0, "augmentation_file": None, "epochs": 2, "batch_size": 16, "effective_batch_size": 16, "learning_rate": 5e-5, "weight_decay": .01, "max_length": 256, "threshold": .5, "train_sha256": sha(BENCH / "train.csv"), "validation_sha256": sha(BENCH / "val.csv"), "origin": "Unchanged upstream DistilBERT baseline script, evaluated safely on real public texts", "upstream_summary": source_summary, "packages": {name: importlib.metadata.version(name) for name in ["torch", "transformers", "datasets", "pandas", "scikit-learn", "accelerate"]}})
history_file = root / "upstream/outputs/distilbert_baseline/trainer"
states = sorted(history_file.glob("checkpoint-*/trainer_state.json"))
runtime = 0
if states:
    logs = json.loads(states[-1].read_text())["log_history"]
    runtime = next((row["train_runtime"] for row in reversed(logs) if "train_runtime" in row), 0)
if runtime == 0:
    import re
    logged = (original / "execution.log").read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"'train_runtime':\s*([0-9.]+)", logged)
    if matches:
        runtime = float(matches[-1])
save_json(out / "status.json", {"status": "complete", "runtime_seconds": runtime, "runtime_note": "training-only runtime from Trainer state; 0 indicates unavailable"})
print(out)
