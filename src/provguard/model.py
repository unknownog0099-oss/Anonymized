from __future__ import annotations

import json
import time
import importlib.metadata
import numpy as np
import pandas as pd
from .core import BENCH, DATA, OUTPUT, ROOT, SETS, environment, integrity, read_texts, save_json, sha, upstream_module

def probabilities(model_dir, texts, batch=16):
    environment()
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir), use_safetensors=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    values = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch):
            enc = tok(texts[start:start+batch], truncation=True, padding=True, max_length=256, return_tensors="pt").to(device)
            values.extend(torch.softmax(model(**enc).logits, dim=-1)[:, 1].cpu().tolist())
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return np.asarray(values)

def evaluate(model_dir, out):
    from pathlib import Path
    from sklearn.metrics import roc_auc_score
    out = Path(out)
    frozen = integrity()
    upstream = upstream_module()
    summary = {"threshold": 0.5, "benchmark_integrity": frozen, "sets": {}}
    for name, relative in SETS.items():
        if name == "train":
            continue
        df = read_texts(BENCH / relative)
        probs = probabilities(model_dir, df.text.tolist())
        y = df.label.astype(int).to_numpy()
        metric = upstream._metrics_at_threshold(probs, y, 0.5)
        # Undefined rates are missing, rather than misleading zeros.
        if not (y == 0).any():
            metric["fpr"] = None
        if not (y == 1).any():
            metric["recall"] = None
        metric["roc_auc"] = float(roc_auc_score(y, probs)) if len(set(y)) == 2 else None
        metric["excluded_missing_n"] = frozen[name]["missing_text_n"]
        metric["groups"] = {}
        for col in ["source", "source_type", "domain", "transform_type", "attack_type", "obfuscation"]:
            if col in df:
                metric["groups"][col] = {str(group): upstream._metrics_at_threshold(probs[rows.index], y[rows.index], 0.5) for group, rows in df.groupby(col)}
        df["true_label"] = y
        df["probability_score"] = probs
        df["predicted_label"] = (probs >= 0.5).astype(int)
        df.to_csv(out / f"{name}_predictions.csv", index=False)
        if name == "test":
            df.loc[(df.true_label == 0) & (df.predicted_label == 1)].to_csv(out / "false_positives.csv", index=False)
            df.loc[(df.true_label == 1) & (df.predicted_label == 0)].to_csv(out / "false_negatives.csv", index=False)
        summary["sets"][name] = metric
    save_json(out / "evaluation.json", summary)
    return summary

def train(method="baseline", seed=42, augment=None, batch=16, accumulation=1, resume=False, semantic_review=None):
    environment()
    import torch
    from datasets import Dataset, DatasetDict
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, set_seed
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; restore GPU environment before research training.")
    if batch * accumulation != 16:
        raise ValueError("Controlled experiment requires effective batch 16.")
    if method == "baseline" and augment:
        raise ValueError("Baseline cannot receive augmentation.")
    if method != "baseline" and not augment:
        raise ValueError("Augmented arm requires an augmentation file.")
    integrity()
    out = OUTPUT / method / f"seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    previous = json.loads((out / 'config.json').read_text()) if (out / 'config.json').exists() else None
    if previous is not None and not resume:
        raise FileExistsError(f"Run already exists: {out}; preserve prior attempt.")
    start = time.monotonic()
    set_seed(seed)
    upstream = upstream_module()
    tr = read_texts(BENCH / "train.csv", strict=True)
    va = read_texts(BENCH / "val.csv", strict=True)
    extra_n = 0
    review_file=DATA/'semantic_review.csv'
    if semantic_review is not None:
        from pathlib import Path
        review_file=Path(semantic_review).resolve()
        if not review_file.is_relative_to(DATA.resolve()):
            raise ValueError('Semantic review must be a versioned project-data artifact.')
    if augment:
        from pathlib import Path
        augment = Path(augment).resolve()
        if BENCH.resolve() in augment.parents:
            raise ValueError("Benchmark files cannot be augmentation inputs.")
        extra = read_texts(augment, strict=True)
        if not extra.label.astype(int).eq(0).all():
            raise ValueError("All hard negatives must be benign.")
        from .leakage import LeakageIndex
        # Reload the canonical persisted representation independently. Never
        # refit IDF on this arm or on the verification batch.
        checked, _ = LeakageIndex.load().filter(extra)
        if len(checked) != len(extra):
            raise ValueError("Augmentation failed independent leakage check.")
        if method.startswith("provguard") or method.startswith("random_real"):
            for col in ["review_A", "reason_A", "review_B", "reason_B", "final_decision", "candidate_id"]:
                if col not in extra:
                    raise ValueError("Real augmentation requires two-pass semantic adjudication.")
            if not (extra.review_A.eq("BENIGN") & extra.review_B.eq("BENIGN") & extra.final_decision.eq("BENIGN")).all() or extra.reason_A.eq("").any() or extra.reason_B.eq("").any():
                raise ValueError("Incomplete or non-benign semantic adjudication.")
            review = pd.read_csv(review_file, keep_default_na=False)
            bound = extra[["candidate_id","text"]].merge(review[["candidate_id","text","final_decision"]],on="candidate_id",suffixes=("","_review"),validate="one_to_one")
            if len(bound)!=len(extra) or not bound.text.eq(bound.text_review).all() or not bound.final_decision.eq("BENIGN").all():
                raise ValueError("Augmentation not bound to exact semantically approved text.")
        extra_n = len(extra)
        if method in {'provguard_dolly_250','provguard_oasst1_250'} and not extra.source.eq(method.split('_')[1]).all():
            raise ValueError('Source-isolated augmentation contains another source corpus.')
        required_n = {"random_real": 1000, "provguard_250": 250, "provguard_500": 500, "provguard_1000": 1000,"provguard_dolly_250":250,"provguard_oasst1_250":250,'random_real_2024':1000,'random_real_7777':1000}.get(method)
        if required_n is not None and extra_n != required_n:
            raise ValueError(f"{method} requires exactly {required_n} audited augmentation rows, found {extra_n}.")
        tr = pd.concat([tr, extra], ignore_index=True)
    config = {"method": method, "seed": seed, "model": "distilbert-base-uncased", "training_rows": len(tr), "augmentation_size": extra_n, "augmentation_file": str(augment) if augment else None, "augmentation_sha256": sha(augment) if augment else None, "train_sha256": sha(BENCH / "train.csv"), "validation_sha256": sha(BENCH / "val.csv"), "epochs": 2, "batch_size": batch, "effective_batch_size": batch*accumulation, "learning_rate": 5e-5, "weight_decay": 0.01, "max_length": 256, "threshold": 0.5, "device": torch.cuda.get_device_name(0), "packages": {name: importlib.metadata.version(name) for name in ["torch", "transformers", "datasets", "pandas", "scikit-learn", "accelerate"]}}
    if previous:
        controlled = ['method','seed','model','training_rows','augmentation_size','augmentation_sha256','train_sha256','validation_sha256','epochs','batch_size','effective_batch_size','learning_rate','weight_decay','max_length','threshold']
        if any(previous.get(k) != config.get(k) for k in controlled):
            raise ValueError('Cannot resume with changed training data/configuration.')
        status = json.loads((out/'status.json').read_text()) if (out/'status.json').exists() else {}
        if status.get('status') == 'complete':
            raise FileExistsError('Completed runs must be reused, never retrained.')
        if (out/'model/model.safetensors').exists():
            evaluate(out/'model',out)
            save_json(out/'status.json',{'status':'complete','runtime_seconds':status.get('runtime_seconds',0)+time.monotonic()-start,'resumed_evaluation':True})
            return
        from transformers.utils import check_torch_load_is_safe
        check_torch_load_is_safe()
    save_json(out / "config.json", config)
    config["upstream_commit"] = "7984e2d6d95e98e914d2b0cd92574b1752614b10"
    config["upstream_script_sha256"] = sha(ROOT / "upstream/src/baselines/distilbert_baseline.py")
    config["source_revision_sha256"] = sha(ROOT / "src/provguard/model.py")
    config.update(max_gradient_norm=1.0, checkpoint_selection='highest original PIDS validation F1',gradient_accumulation_steps=accumulation)
    cache=ROOT/'.cache/huggingface/hub/models--distilbert-base-uncased'
    reference=cache/'refs/main'
    pretrained=config['model']
    if reference.exists():
        revision=reference.read_text().strip()
        snapshot=cache/'snapshots'/revision
        if (snapshot/'model.safetensors').exists():
            pretrained=str(snapshot)
            config.update(pretrained_revision=revision,pretrained_sha256=sha(snapshot/'model.safetensors'),pretrained_tokenizer_sha256={p.name:sha(p) for p in snapshot.glob('*') if p.is_file() and p.name!='model.safetensors'})
    # Preserve exact local implementation bytes alongside the source revision.
    source_dir=out/'source'
    source_dir.mkdir(exist_ok=True)
    for path in sorted((ROOT/'src/provguard').glob('*.py')):
        target=source_dir/path.name
        if not target.exists():
            target.write_bytes(path.read_bytes())
    if augment:
        config["leakage_index_sha256"] = sha(DATA / "leakage_index_v2/metadata.json")
        if method.startswith('provguard') or method.startswith('random_real'):
            config['semantic_review_file']=str(review_file)
            config['semantic_review_sha256']=sha(review_file)
        else:
            config['semantic_review_sha256']=None
    save_json(out / "config.json", config)
    save_json(out / "status.json", {"status": "running"})
    try:
        ds = DatasetDict({"train": Dataset.from_pandas(tr[["text", "label"]], preserve_index=False), "validation": Dataset.from_pandas(va[["text", "label"]], preserve_index=False)})
        tok = AutoTokenizer.from_pretrained(pretrained)
        tokenized = upstream._tokenize_splits(ds, tok)
        model = AutoModelForSequenceClassification.from_pretrained(pretrained, num_labels=2, use_safetensors=True)
        args = TrainingArguments(output_dir=str(out / "trainer"), eval_strategy="epoch", save_strategy="epoch", learning_rate=5e-5, per_device_train_batch_size=batch, per_device_eval_batch_size=batch, gradient_accumulation_steps=accumulation, num_train_epochs=2, weight_decay=0.01, load_best_model_at_end=True, metric_for_best_model="f1", greater_is_better=True, logging_steps=50, report_to=[], seed=seed, save_total_limit=1, max_grad_norm=1.0)
        config.update(optimizer=args.optim.value,lr_scheduler=args.lr_scheduler_type.value,warmup_steps=args.warmup_steps,fp16=args.fp16,bf16=args.bf16)
        save_json(out/'config.json',config)
        trainer = Trainer(model=model, args=args, train_dataset=tokenized["train"], eval_dataset=tokenized["validation"], data_collator=DataCollatorWithPadding(tokenizer=tok), compute_metrics=upstream._compute_metrics)
        checkpoint = None
        if resume and previous:
            checkpoints = sorted((out/'trainer').glob('checkpoint-*'),key=lambda p:int(p.name.split('-')[-1]))
            checkpoints = [p for p in checkpoints if (p/'trainer_state.json').exists() and (p/'optimizer.pt').exists() and (p/'scheduler.pt').exists()]
            if not checkpoints:
                raise ValueError('No safe optimizer/trainer checkpoint for resuming.')
            checkpoint = str(checkpoints[-1])
        trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model(str(out / "model"))
        tok.save_pretrained(str(out / "model"))
        save_json(out / "trainer_history.json", trainer.state.log_history)
        del trainer, model
        torch.cuda.empty_cache()
        evaluate(out / "model", out)
        save_json(out / "status.json", {"status": "complete", "runtime_seconds": time.monotonic()-start})
    except Exception as exc:
        save_json(out / "status.json", {"status": "failed", "error": repr(exc), "runtime_seconds": time.monotonic()-start})
        raise
    integrity()

def score(seeds=(13,42,123)):
    if set(seeds) != {13,42,123} or len(seeds)!=3:
        raise ValueError('Corrected primary selection requires the fixed unaugmented selector seeds 13,42,123.')
    df = pd.read_csv(DATA / "clean_candidates.csv", keep_default_na=False)
    log = json.loads((ROOT / "reports/leakage_filter.json").read_text())
    if sha(DATA / "clean_candidates.csv") != log["output_sha256"]:
        raise ValueError("Candidate pool changed after leakage filtering.")
    for seed in seeds:
        path = OUTPUT / "baseline" / f"seed_{seed}"
        config = json.loads((path / "config.json").read_text())
        status = json.loads((path / "status.json").read_text())
        if config["augmentation_size"] != 0 or status["status"] != "complete":
            raise ValueError("Selector must be a completed untouched baseline arm.")
        df[f"attack_probability_seed_{seed}"] = probabilities(path / "model", df.text.tolist())
    df["attack_probability"] = df[[f"attack_probability_seed_{s}" for s in seeds]].mean(axis=1)
    df["baseline_seed"] = ",".join(map(str, seeds))
    for s in seeds:
        df[f'score_seed{s}']=df[f'attack_probability_seed_{s}']
    df['mean_attack_probability']=df.attack_probability
    df=df.sort_values(['mean_attack_probability','candidate_id'],ascending=[False,True]).reset_index(drop=True)
    df['rank']=np.arange(1,len(df)+1)
    ordered=df.sort_values(['char_length','candidate_id'])
    bins=pd.Series(np.minimum(np.arange(len(ordered))*4//len(ordered),3),index=ordered.candidate_id)
    df['length_bin']=df.candidate_id.map(bins)
    df.to_csv(DATA / "scored_candidates.csv", index=False)
