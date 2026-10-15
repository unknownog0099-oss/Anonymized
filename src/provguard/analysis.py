from __future__ import annotations
import json
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from .core import ROOT, OUTPUT, save_json

METRICS = ["iid_f1", "iid_precision", "iid_recall", "iid_fpr", "roc_auc", "hard_benign_fpr", "external_hb_fpr", "curated_hb_fpr", "obfuscated_recall", "domain_ood_f1", "structural_ood_f1", "structural_ood_fpr"]

def paired(baseline, treatment, repetitions=10000, seed=42):
    keys = ["eval_id", "true_label", "probability_score"]
    if any(c not in baseline or c not in treatment for c in keys):
        raise ValueError("Paired predictions require stable IDs, labels, probabilities.")
    if baseline.eval_id.duplicated().any() or treatment.eval_id.duplicated().any() or set(baseline.eval_id) != set(treatment.eval_id):
        raise ValueError("Paired evaluation IDs differ or are duplicated.")
    pair = baseline[keys].merge(treatment[keys], on="eval_id", suffixes=("_b", "_p"), validate="one_to_one")
    if not pair.true_label_b.equals(pair.true_label_p):
        raise ValueError("Paired labels differ.")
    if not pair.true_label_b.eq(0).all() or pair.empty:
        raise ValueError("FPR statistics require a nonempty benign-only subset.")
    b = pair.probability_score_b.to_numpy() >= .5
    p = pair.probability_score_p.to_numpy() >= .5
    fixed = int((b & ~p).sum())
    broken = int((~b & p).sum())
    delta = p.astype(float) - b.astype(float)
    rng = np.random.default_rng(seed)
    # Multinomial counts are equivalent to resampling the paired per-row deltas.
    counts = np.array([(delta == -1).sum(), (delta == 0).sum(), (delta == 1).sum()])
    draws = rng.multinomial(len(delta), counts / len(delta), size=repetitions)
    boot = (draws[:, 2] - draws[:, 0]) / len(delta)
    return {"n": len(delta), "baseline_fp_fixed": fixed, "new_fp": broken, "fpr_delta": float(delta.mean()), "bootstrap_95_ci": np.quantile(boot, [.025, .975]).tolist(), "mcnemar_exact_p": float(binomtest(min(fixed, broken), fixed+broken, .5).pvalue) if fixed+broken else 1.0}

def aggregate():
    from .final_analysis import export_results
    return export_results(final=False)


def plot():
    from .final_analysis import export_results
    return export_results(final=True)
