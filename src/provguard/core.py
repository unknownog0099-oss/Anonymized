from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "upstream"
BENCH = UPSTREAM / "data/pids_bench_v3"
DATA = ROOT / "data/provguard"
OUTPUT = ROOT / "outputs/provguard"
SETS = {
    "train": "train.csv", "val": "val.csv", "test": "test.csv",
    "hard_benign": "eval_subsets/hard_benign_test.csv",
    "balanced_subtype": "eval_subsets/balanced_subtype_test.csv",
    "obfuscated": "eval_subsets/obfuscated_attacks.csv",
    "domain_ood": "ood/domain_ood.csv", "structural_ood": "ood/structural_ood.csv",
}

def environment():
    # Keep downloads and caches within the authorized project folder.
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def normalize(text):
    return " ".join(text.lower().split()) if isinstance(text, str) else ""

def read_texts(path, strict=False):
    import pandas as pd
    df = pd.read_csv(path, keep_default_na=False)
    if "text" not in df:
        raise ValueError(f"Missing text column: {path}")
    mask = df.text.map(lambda x: isinstance(x, str) and bool(x.strip()) and x.strip().lower() not in {"nan", "null"})
    if strict and not mask.all():
        raise ValueError(f"Missing/corrupt training text: {path}")
    df = df.loc[mask].copy()
    df["eval_id"] = [f"{Path(path).stem}:{i}" for i in df.index]
    return df.reset_index(drop=True)

def integrity():
    import re
    manifest = (BENCH / "FREEZE_MANIFEST.txt").read_text(encoding="utf-8")
    report = {}
    for name, relative in SETS.items():
        path = BENCH / relative
        stem = path.stem
        match = re.search(rf"^{stem}\s+\d+\s+\d+\s+\d+\s+([a-f0-9]{{64}})", manifest, re.M)
        if not match or sha(path) != match[1]:
            raise ValueError(f"Frozen checksum mismatch: {path}; restore exact upstream Git bytes (disable autocrlf).")
        df = read_texts(path)
        import pandas as pd
        raw = pd.read_csv(path, keep_default_na=False)
        report[name] = {"sha256": sha(path), "rows": len(raw), "usable_text_n": len(df), "missing_text_n": len(raw)-len(df)}
    save_json(ROOT / "reports/benchmark_integrity.json", report)
    return report

def upstream_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("pids_distilbert", UPSTREAM / "src/baselines/distilbert_baseline.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
