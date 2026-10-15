from __future__ import annotations

import hashlib
import sys
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from .core import BENCH, DATA, ROOT, SETS, UPSTREAM, environment, integrity, normalize, read_texts, save_json, sha

def download():
    environment()
    from datasets import load_dataset
    rows = []
    metadata = {}
    for name, source in [("databricks/databricks-dolly-15k", "dolly"), ("OpenAssistant/oasst1", "oasst1")]:
        dataset = load_dataset(name, split="train")
        metadata[source] = {"dataset": name, "fingerprint": dataset._fingerprint, "raw_n": len(dataset), "split": "train"}
        for i, row in enumerate(dataset):
            if source == "oasst1" and (row.get("role") != "prompter" or row.get("lang") != "en" or row.get("deleted") or row.get("parent_id") is not None):
                continue
            text = row.get("instruction" if source == "dolly" else "text")
            if not isinstance(text, str) or len(text.strip()) < 20 or "\ufffd" in text or "\x00" in text:
                continue
            original = str(row.get("message_id", i))
            cid = hashlib.sha256(f"{source}:{original}".encode()).hexdigest()[:20]
            rows.append({"candidate_id": cid, "text": text, "source": source, "original_id": original, "char_length": len(text)})
    DATA.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df = df.loc[~df.text.map(normalize).duplicated()].reset_index(drop=True)
    df.to_csv(DATA / "real_benign_candidates.csv", index=False)
    metadata["clean_n"] = len(df)
    metadata["source_n"] = df.source.value_counts().to_dict()
    save_json(ROOT / "reports/candidate_download.json", metadata)
    print(f"Saved {len(df)} candidate prompts; labels remain provisional until audited.")

def leakage_filter(candidates, references, threshold=0.90, batch=128, index=None):
    """Use a fixed persisted index; a different reference pool must never refit IDF."""
    from .leakage import LeakageIndex, corpus_hash
    index = LeakageIndex.load() if index is None else index
    ref_hash=corpus_hash([normalize(x) for x in references if normalize(x)])
    if ref_hash != index.metadata['reference_text_sha256']:
        raise ValueError('References differ from the frozen leakage representation; do not refit on a candidate batch.')
    return index.filter(candidates,threshold,batch)


def dedup():
    frozen = integrity()
    reference = []
    for relative in SETS.values():
        reference.extend(read_texts(BENCH / relative).text.tolist())
    df = pd.read_csv(DATA / "real_benign_candidates.csv", keep_default_na=False)
    clean, log = leakage_filter(df, reference)
    clean.to_csv(DATA / "clean_candidates.csv", index=False)
    log["references"] = frozen
    log["input_sha256"] = sha(DATA / "real_benign_candidates.csv")
    log["output_sha256"] = sha(DATA / "clean_candidates.csv")
    save_json(ROOT / "reports/leakage_filter.json", log)
    print(log)

def audit_queue():
    df = pd.read_csv(DATA / "scored_candidates.csv", keep_default_na=False)
    df = df.sort_values(["attack_probability", "candidate_id"], ascending=[False, True])
    # All candidates are included so reviewers can audit replacements and random controls.
    for col in ["decision", "reason", "reviewer"]:
        df[col] = ""
    path = DATA / "audit.csv"
    if path.exists():
        raise FileExistsError("Audit exists; preserve completed reviews.")
    df.to_csv(path, index=False)
    print(f"Review {path}: decision keep/remove/ambiguous, with reason and reviewer.")

def audited_pool(scored, audit):
    if audit.candidate_id.duplicated().any() or scored.candidate_id.duplicated().any():
        raise ValueError("Duplicate candidate IDs.")
    merged = scored.merge(audit[["candidate_id", "text", "decision", "reason", "reviewer"]], on="candidate_id", suffixes=("", "_audit"), validate="one_to_one")
    if not (merged.text == merged.text_audit).all():
        raise ValueError("Audit text does not match scored text.")
    allowed = {"", "keep", "remove", "ambiguous"}
    if not set(merged.decision).issubset(allowed):
        raise ValueError("Invalid audit decision.")
    approved = merged.decision.eq("keep")
    if (merged.loc[approved, ["reason", "reviewer"]].eq("").any(axis=1)).any():
        raise ValueError("Each keep decision requires reason and reviewer.")
    return merged.loc[approved].drop(columns="text_audit").copy()

def matched_random(pool, hard, seed=42):
    """Draw without replacement within identical source/pooled length-quartile strata."""
    pool = pool.copy()
    pool["length_bin"] = pd.qcut(pool.char_length.rank(method="first"), 4, labels=False)
    hard = hard.merge(pool[["candidate_id", "length_bin"]], on="candidate_id", validate="one_to_one")
    samples = []
    for (source, length_bin), group in hard.groupby(["source", "length_bin"]):
        eligible = pool.loc[pool.source.eq(source) & pool.length_bin.eq(length_bin)]
        if len(eligible) < len(group):
            raise ValueError("Insufficient reviewed random control stratum.")
        samples.append(eligible.sample(n=len(group), random_state=seed))
    # Independent arms can share examples; excluding hard examples would bias the control.
    return pd.concat(samples, ignore_index=True)

def build_sets():
    from .semantic import build_final_sets
    return build_final_sets()


def curated():
    integrity()
    sys.path.insert(0, str(UPSTREAM))
    from data_builder import build_hard_negative as builder
    scratch = DATA / "curated_builder"
    scratch.mkdir(parents=True, exist_ok=True)
    builder.OUT_DIR = scratch
    builder.build(train_size=1000, val_size=200, near_dup_threshold=0.90, seed=42)
    from .leakage import prepare
    prepare()
    log = __import__('json').loads((ROOT/'reports/canonical_leakage_audit.json').read_text())['curated']
    log['limitation']='All canonical-valid unique builder train rows are used without duplication; this arm is not size-controlled.'
    save_json(ROOT/'reports/curated_builder.json',log)
