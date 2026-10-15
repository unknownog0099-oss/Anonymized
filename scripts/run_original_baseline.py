"""Execute upstream unchanged; raw HB output is invalid because text is withheld.

Retain raw reproduction for diagnostics, then evaluate the saved model on usable
texts separately. Never use raw withheld-text HB scores in paper tables.
"""
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))
from provguard.core import environment, integrity, save_json, sha

environment()
integrity()
out = root / "outputs/original_baseline"
if (out / "status.json").exists():
    raise FileExistsError("Original baseline already attempted; preserve it and inspect its status before retrying.")
out.mkdir(parents=True, exist_ok=True)
script = root / "upstream/src/baselines/distilbert_baseline.py"
save_json(out / "metadata.json", {"script_sha256": sha(script), "seed": 42, "warning": "Raw upstream hard-benign metrics include missing texts and are NOT valid research results."})
with (out / "execution.log").open("w", encoding="utf-8") as log:
    result = subprocess.run([sys.executable, str(script), "--epochs", "2", "--batch", "16", "--lr", "5e-5", "--seed", "42"], cwd=root / "upstream", stdout=log, stderr=subprocess.STDOUT)
save_json(out / "status.json", {"returncode": result.returncode})
if result.returncode:
    raise SystemExit(result.returncode)
from provguard.model import evaluate
evaluate(root / "upstream/models/distilbert", out)
integrity()
