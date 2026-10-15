# ProvGuard: observed research summary

This study tests confusion-selected real benign augmentation against unaugmented, curated and source/length-matched random-real controls using the same DistilBERT and frozen PIDS splits.

Primary threshold 0.5; K=1000. Exact Ns and all individual runs are in results_master.csv. Full-public hard-benign coverage is 808 usable texts with 664 withheld rows excluded. Stress-set FPR is not a deployment prevalence estimate.

Completed matched primary seeds: [].

The four-arm matched comparison is not yet complete; no method-success conclusion is available.

Limitations: public external-text coverage is incomplete; the curated builder may yield fewer examples than the real arms; deterministic validity review is explicitly distinguished from human review and can exclude complex/attack-associated benign wording; unreviewed error categories remain pending. Results cannot establish intent understanding. Direct-review details, actual augmentation counts and selector seeds must accompany publication claims.
