"""Recover a completed upstream run that only failed printing a Unicode arrow.

No training failure is accepted. Requires completed summary/model and a specific
terminal-encoding error after the upstream completion message.
"""
import json
import sys
from pathlib import Path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))
from provguard.core import environment, integrity, save_json
from provguard.model import evaluate
environment()
integrity()
out = root / "outputs/original_baseline"
status = json.loads((out / "status.json").read_text())
if status["returncode"] != 0:
    log = (out / "execution.log").read_text(encoding="utf-8", errors="replace")
    if "UnicodeEncodeError" not in log or "DistilBERT training complete." not in log:
        raise RuntimeError("Not an encoding-only completion failure; investigate original log.")
    summary = json.loads((root / "upstream/outputs/distilbert_baseline/summary.json").read_text())
    if summary["hyperparams"]["seed"] != 42 or summary["hyperparams"]["epochs"] != 2:
        raise RuntimeError("Original completed summary does not match protocol.")
    if not (root / "upstream/models/distilbert/model.safetensors").exists():
        raise RuntimeError("Completed model unavailable.")
    status["original_returncode"] = status["returncode"]
    status["returncode"] = 0
    status["recovery"] = "Training and raw evaluations completed; final arrow status print failed under Windows cp1252. Saved artifacts retained unchanged."
    save_json(out / "status.json", status)
if not (out / "evaluation.json").exists():
    evaluate(root / "upstream/models/distilbert", out)
integrity()
