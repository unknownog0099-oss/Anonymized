"""Restartable corrected study. Completed runs are verified, never retrained."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .core import ROOT, DATA, OUTPUT, BENCH, SETS, environment, integrity, read_texts, sha, save_json

SEEDS = [13, 42, 123, 2024, 7777]
PRIMARY = ['baseline', 'curated', 'random_real', 'provguard_1000']
METHODS = PRIMARY + ['provguard_250', 'provguard_500']
PLAN = [(m, s) for m in METHODS for s in SEEDS]
REGISTRY = ROOT / 'reports/experiment_registry.json'

def now():
    return datetime.now(timezone.utc).isoformat()

def augmentation(method):
    return None if method == 'baseline' else DATA / ('random_real_1000.csv' if method == 'random_real' else f'{method}_hn.csv' if method == 'curated' else f'{method}.csv')

def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def verify_completed(method, seed):
    """Check exact frozen row IDs/text/labels, scores and binding of each run."""
    import numpy as np
    import pandas as pd
    out = OUTPUT / method / f'seed_{seed}'
    if not (out / 'status.json').exists() or load(out/'status.json').get('status') != 'complete':
        return False
    c, e = load(out/'config.json'), load(out/'evaluation.json')
    expected = {'method': method, 'seed': seed, 'model': 'distilbert-base-uncased', 'epochs': 2,
                'effective_batch_size':16, 'learning_rate':5e-5, 'weight_decay':.01, 'max_length':256, 'threshold':.5,
                'train_sha256':sha(BENCH/'train.csv'), 'validation_sha256':sha(BENCH/'val.csv')}
    if any(c.get(k) != v for k,v in expected.items()) or e.get('threshold') != .5:
        raise ValueError(f'Completed run configuration differs: {method}/{seed}')
    if not (out/'model/model.safetensors').is_file():
        raise ValueError(f'Completed run model missing: {out}')
    from safetensors import safe_open
    with safe_open(str(out/'model/model.safetensors'),framework='np') as weights:
        if 'classifier.weight' not in weights.keys() or weights.get_slice('classifier.weight').get_shape() != [2,768]:
            raise ValueError(f'Completed classifier is invalid: {out}')
    aug = augmentation(method)
    if aug:
        if c.get('augmentation_sha256') != sha(aug) or c.get('leakage_index_sha256') != sha(DATA/'leakage_index_v2/metadata.json'):
            raise ValueError(f'Completed augmentation binding differs: {out}')
        review_file=Path(c.get('semantic_review_file',DATA/'semantic_review.csv'))
        if method != 'curated' and c.get('semantic_review_sha256') != sha(review_file):
            raise ValueError(f'Completed semantic review binding differs: {out}')
    elif c.get('augmentation_size') != 0:
        raise ValueError('Baseline has augmentation.')
    upstream = None
    for name, relative in SETS.items():
        if name == 'train':
            continue
        actual = pd.read_csv(out/f'{name}_predictions.csv', keep_default_na=False)
        reference = read_texts(BENCH/relative)
        if actual.eval_id.duplicated().any() or len(actual) != len(reference):
            raise ValueError(f'Wrong prediction coverage: {out}/{name}')
        joined = reference[['eval_id','text','label']].merge(actual,on='eval_id',suffixes=('_frozen',''),validate='one_to_one')
        if len(joined)!=len(reference) or not joined.text_frozen.eq(joined.text).all() or not joined.label_frozen.astype(int).eq(joined.true_label.astype(int)).all():
            raise ValueError(f'Prediction text or label differs: {out}/{name}')
        p = joined.probability_score.to_numpy(dtype=float)
        if not np.isfinite(p).all() or ((p<0)|(p>1)).any() or not np.array_equal((p>=.5).astype(int),joined.predicted_label.astype(int)):
            raise ValueError(f'Invalid probabilities/threshold: {out}/{name}')
        if e['benchmark_integrity'][name]['sha256'] != sha(BENCH/relative) or e['sets'][name]['n'] != len(reference):
            raise ValueError(f'Evaluation provenance differs: {out}/{name}')
    return True

def running_training():
    import psutil
    found=[]
    for proc in psutil.process_iter(['pid','name']):
        if 'python' not in (proc.info.get('name') or '').lower():
            continue
        try:
            args=proc.cmdline()
            if any('provguard.py' in a for a in args) and 'train' in args:
                found.append({'pid':proc.pid,'command':args})
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
    return found

def snapshot_incomplete(out):
    # Resolve and verify both targets; preserve failed attempts rather than delete.
    destination = ROOT/'outputs/failed_attempts'/out.parent.name/(out.name+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    for path in [out.resolve(),destination.resolve()]:
        if not path.is_relative_to(ROOT.resolve()) or path == ROOT.resolve():
            raise ValueError('Archive target outside workspace.')
    destination.parent.mkdir(parents=True,exist_ok=True)
    shutil.move(str(out),str(destination))
    return str(destination)

def safe_checkpoint(out):
    checkpoints=list((out/'trainer').glob('checkpoint-*/trainer_state.json'))
    if not checkpoints:
        return False
    # Current torch 2.5 can load safetensors models, but Transformers forbids
    # unsafe optimizer deserialization. Preserve and restart that attempt.
    from transformers.utils import check_torch_load_is_safe
    try:
        check_torch_load_is_safe()
    except ValueError:
        return False
    return any((p.parent/'optimizer.pt').exists() and (p.parent/'scheduler.pt').exists() for p in checkpoints)

def initialize():
    previous=load(REGISTRY) if REGISTRY.exists() else {}
    prior={f"{r['method']}/{r['seed']}":r for r in previous.get('experiments',[])}
    rows=[]
    for method,seed in PLAN:
        row=prior.get(f'{method}/{seed}',{'method':method,'seed':seed,'status':'PENDING','started_at':None,'completed_at':None,'failure_reason':None,'attempts':[]})
        row['output_path']=str(OUTPUT/method/f'seed_{seed}')
        if verify_completed(method,seed):
            model_hash=sha(Path(row['output_path'])/'model/model.safetensors')
            if row.get('model_sha256') and row['model_sha256'] != model_hash:
                raise ValueError(f'Completed model changed after registry verification: {method}/{seed}')
            row['model_sha256']=model_hash
            row.update(status='COMPLETED',failure_reason=None)
            row['completed_at']=row['completed_at'] or now()
            row['reused_existing']=True
        elif row['status']=='RUNNING' and not running_training():
            row.update(status='FAILED',failure_reason='Interrupted process; checkpoint will be checked before restart.')
        rows.append(row)
    return {'version':2,'protocol':'corrected canonical leakage + independent semantic adjudication','updated_at':now(),'experiments':rows}

def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--inspect',action='store_true')
    args=parser.parse_args(argv)
    environment()
    os.environ['PYTHONIOENCODING']='utf-8'
    integrity()
    if args.inspect:
        save_json(REGISTRY,initialize())
        return
    import psutil
    lock=ROOT/'reports/study_controller.json'
    if lock.exists():
        old=load(lock)
        if old.get('status')=='running' and old.get('pid') != os.getpid() and psutil.pid_exists(old.get('pid',0)):
            try:
                if any('finish_study.py' in a or 'run_full_study.py' in a or 'bootstrap_full_study.py' in a for a in psutil.Process(old['pid']).cmdline()):
                    raise RuntimeError(f"Study controller already running: PID {old['pid']}")
            except psutil.NoSuchProcess:
                pass
    state={'pid':os.getpid(),'status':'running','started_at':now()}
    save_json(lock,state)
    registry=initialize()
    def persist():
        registry['updated_at']=now()
        save_json(REGISTRY,registry)
        save_json(lock,state)
    persist()
    logs=ROOT/'outputs/study_logs_v2'
    logs.mkdir(parents=True,exist_ok=True)
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required for the controlled study.')
        save_json(ROOT/'reports/environment_report.json',{'created_at':now(),'python':sys.version,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'packages':{d.metadata['Name']:d.version for d in __import__('importlib.metadata',fromlist=['distributions']).distributions()},'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'src/provguard').glob('*.py'))},'upstream_commit':'7984e2d6d95e98e914d2b0cd92574b1752614b10'})
        # Adopt the previously launched baseline without competing on the GPU.
        while running_training():
            state['waiting_for_existing_training']=running_training()
            persist()
            time.sleep(15)
        state.pop('waiting_for_existing_training',None)
        real_ready=False
        for row in registry['experiments']:
            method,seed=row['method'],row['seed']
            if verify_completed(method,seed):
                row.update(status='COMPLETED',completed_at=row['completed_at'] or now(),failure_reason=None)
                persist()
                continue
            if method not in ['baseline','curated'] and not real_ready:
                # Both review passes must finish. This never falls back to the old easy sets.
                from .semantic import build_final_sets
                state['current_stage']='verify_semantic_adjudication_and_build_sets'
                persist()
                build_final_sets()
                real_ready=True
            out=Path(row['output_path'])
            cmd=[sys.executable,str(ROOT/'scripts/provguard.py'),'train','--method',method,'--seed',str(seed)]
            aug=augmentation(method)
            if aug:
                cmd += ['--augment-train',str(aug)]
            if out.exists() and any(out.iterdir()):
                if (out/'config.json').exists() and ((out/'model/model.safetensors').exists() or safe_checkpoint(out)):
                    cmd += ['--resume']
                else:
                    archived=snapshot_incomplete(out)
                    row['attempts'].append({'archived_path':archived,'reason':'No safe resumable checkpoint; old failure preserved.','at':now()})
            logpath=logs/f'{method}_seed_{seed}_{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")}.log'
            row.update(status='RUNNING',started_at=now(),failure_reason=None,log_path=str(logpath),command=cmd)
            state.update(current_stage='training',current_run=f'{method}/{seed}')
            persist()
            with logpath.open('w',encoding='utf-8') as log:
                process=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy())
                row['pid']=process.pid
                persist()
                code=process.wait()
            row['attempts'].append({'command':cmd,'log_path':str(logpath),'returncode':code,'ended_at':now()})
            row.pop('pid',None)
            if code or not verify_completed(method,seed):
                row.update(status='FAILED',failure_reason=f'Training/evaluation failed, return code {code}; inspect {logpath}')
                persist()
                raise RuntimeError(row['failure_reason'])
            row.update(status='COMPLETED',completed_at=now())
            row['model_sha256']=sha(out/'model/model.safetensors')
            persist()
            from .final_analysis import export_results
            export_results(final=False)
        from .final_analysis import export_results
        state['current_stage']='final_statistics_and_figures'
        persist()
        export_results(final=True)
        state.update(status='complete',completed_at=now())
        persist()
    except Exception as exc:
        state.update(status='failed',error=repr(exc),failed_at=now())
        persist()
        raise

