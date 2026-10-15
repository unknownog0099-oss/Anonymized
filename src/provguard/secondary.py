"""Secondary confirmations, strictly after the thirty-run DistilBERT study."""
from __future__ import annotations
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import pandas as pd
from .core import ROOT,DATA,OUTPUT,BENCH,SETS,environment,integrity,read_texts,sha,save_json,upstream_module
from .study import PLAN,verify_completed,snapshot_incomplete
from .final_analysis import align,rates,crossed_bootstrap,md,csv

SECONDARY_SEEDS=[13,42,123]
MODEL='microsoft/deberta-v3-base'

def outpath(family,method,seed):
    return ROOT/'outputs/secondary/deberta'/method/f'seed_{seed}' if family=='deberta' else OUTPUT/method/f'seed_{seed}'

def approved(frame):
    from .leakage import LeakageIndex
    if not frame.label.astype(int).eq(0).all() or frame.candidate_id.duplicated().any():
        raise ValueError('Secondary augmentation must be unique label-zero examples.')
    for c in ['review_A','review_B','final_decision']:
        if not frame[c].eq('BENIGN').all():
            raise ValueError('Secondary augmentation lacks two BENIGN decisions.')
    if any(frame[c].astype(str).str.strip().eq('').any() for c in ['reason_A','reason_B']):
        raise ValueError('Secondary semantic reasons are missing.')
    review=pd.read_csv(DATA/'semantic_review.csv',keep_default_na=False)
    p=frame[['candidate_id','text']].merge(review[['candidate_id','text','final_decision']],on='candidate_id',suffixes=('','_review'),validate='one_to_one')
    if len(p)!=len(frame) or not p.text.eq(p.text_review).all() or not p.final_decision.eq('BENIGN').all():
        raise ValueError('Secondary augmentation text is not bound to original semantic approval.')
    checked,_=LeakageIndex.load().filter(frame)
    if len(checked)!=len(frame):
        raise ValueError('Secondary augmentation failed canonical leakage check.')

def complete(family,method,seed,augment):
    if family=='source':
        return verify_completed(method,seed)
    out=outpath(family,method,seed)
    if not (out/'status.json').exists() or json.loads((out/'status.json').read_text()).get('status')!='complete':
        return False
    c=json.loads((out/'config.json').read_text())
    expected={'method':method,'seed':seed,'model':MODEL,'epochs':2,'max_length':256,'learning_rate':5e-5,'weight_decay':.01,'effective_batch_size':16,'batch_size':4,'gradient_accumulation_steps':4,'threshold':.5,'train_sha256':sha(BENCH/'train.csv'),'validation_sha256':sha(BENCH/'val.csv'),'augmentation_sha256':sha(augment) if augment else None}
    if any(c.get(k)!=v for k,v in expected.items()) or not (out/'model/model.safetensors').exists():
        raise ValueError('Completed DeBERTa configuration/artifacts differ.')
    if augment and (c.get('semantic_review_sha256')!=sha(DATA/'semantic_review.csv') or c.get('leakage_index_sha256')!=sha(DATA/'leakage_index_v2/metadata.json')):
        raise ValueError('DeBERTa augmentation review/index binding differs.')
    evaluation=json.loads((out/'evaluation.json').read_text())
    if evaluation.get('threshold')!=.5:
        raise ValueError('DeBERTa evaluation threshold differs.')
    for name,relative in SETS.items():
        if name=='train':
            continue
        reference=read_texts(BENCH/relative)
        actual=pd.read_csv(out/f'{name}_predictions.csv',keep_default_na=False)
        p=reference[['eval_id','text','label']].merge(actual,on='eval_id',suffixes=('_frozen',''),validate='one_to_one')
        if len(p)!=len(reference) or len(actual)!=len(reference) or not p.text_frozen.eq(p.text).all() or not p.label_frozen.astype(int).eq(p.true_label.astype(int)).all():
            raise ValueError('Completed DeBERTa prediction coverage/text differs.')
        probabilities=p.probability_score.to_numpy(dtype=float)
        if not np.isfinite(probabilities).all() or ((probabilities<0)|(probabilities>1)).any() or not np.array_equal((probabilities>=.5).astype(int),p.predicted_label.astype(int)):
            raise ValueError('Invalid DeBERTa scores or prediction threshold.')
        if evaluation['benchmark_integrity'][name]['sha256']!=sha(BENCH/relative) or evaluation['sets'][name]['n']!=len(reference):
            raise ValueError('DeBERTa evaluation integrity differs.')
    return True

def train_deberta(method,seed,augment=None):
    environment()
    import torch
    from datasets import Dataset,DatasetDict
    from transformers import AutoTokenizer,AutoModelForSequenceClassification,TrainingArguments,Trainer,DataCollatorWithPadding,set_seed
    from .model import evaluate
    integrity()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable for secondary confirmation.')
    if method not in ['baseline','provguard_1000'] or (method=='baseline' and augment) or (method=='provguard_1000' and not augment):
        raise ValueError('Invalid secondary arm.')
    out=outpath('deberta',method,seed)
    if complete('deberta',method,seed,augment):
        return
    if out.exists() and any(out.iterdir()):
        if (out/'model/model.safetensors').exists():
            previous=json.loads((out/'config.json').read_text())
            if previous.get('augmentation_sha256')!=(sha(augment) if augment else None):
                raise ValueError('Cannot resume secondary evaluation with changed augmentation.')
            evaluate(out/'model',out)
            save_json(out/'status.json',{'status':'complete','runtime_seconds':None,'runtime_note':'Original interrupted wall time unavailable; resumed saved-model evaluation.'})
            return
        archived=snapshot_incomplete(out)
        save_json(ROOT/f'reports/secondary_restart_{method}_{seed}.json',{'archive':archived,'reason':'No completed model; safe optimizer restore unavailable in installed Torch; original attempt retained.'})
    out.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    config={'method':method,'seed':seed,'model':MODEL,'max_length':256,'epochs':2,'batch_size':4,'gradient_accumulation_steps':4,'effective_batch_size':16,'learning_rate':5e-5,'weight_decay':.01,'max_gradient_norm':1.,'threshold':.5,'checkpoint_selection':'highest original PIDS validation F1','train_sha256':sha(BENCH/'train.csv'),'validation_sha256':sha(BENCH/'val.csv'),'augmentation_size':1000 if augment else 0,'augmentation_file':str(augment) if augment else None,'augmentation_sha256':sha(augment) if augment else None,'semantic_review_sha256':sha(DATA/'semantic_review.csv') if augment else None,'leakage_index_sha256':sha(DATA/'leakage_index_v2/metadata.json') if augment else None,'source_revision_sha256':sha(Path(__file__)),'rationale':'Same fixed main-study hyperparameters; microbatch 4 and accumulation 4 predefined for both DeBERTa arms on the 8 GiB GPU; no method-specific tuning.'}
    save_json(out/'config.json',config)
    save_json(out/'status.json',{'status':'running'})
    try:
        set_seed(seed)
        tr=read_texts(BENCH/'train.csv',strict=True)
        va=read_texts(BENCH/'val.csv',strict=True)
        if augment:
            extra=read_texts(augment,strict=True)
            if len(extra)!=1000:
                raise ValueError('DeBERTa ProvGuard requires 1000 verified examples.')
            approved(extra)
            tr=pd.concat([tr,extra],ignore_index=True)
        # Pin the first downloaded official revision for every confirmation run.
        revision_file=ROOT/'reports/deberta_pretrained_revision.json'
        revision=json.loads(revision_file.read_text())['revision'] if revision_file.exists() else None
        tok=AutoTokenizer.from_pretrained(MODEL,revision=revision)
        model=AutoModelForSequenceClassification.from_pretrained(MODEL,revision=revision,num_labels=2,use_safetensors=True)
        revision=model.config._commit_hash
        if not revision:
            raise ValueError('Cannot determine and freeze DeBERTa pretrained revision.')
        if revision_file.exists() and revision!=json.loads(revision_file.read_text())['revision']:
            raise ValueError('DeBERTa pretrained revision changed.')
        save_json(revision_file,{'model':MODEL,'revision':revision})
        config['pretrained_revision']=revision
        ds=DatasetDict({'train':Dataset.from_pandas(tr[['text','label']],preserve_index=False),'validation':Dataset.from_pandas(va[['text','label']],preserve_index=False)})
        tokenized=upstream_module()._tokenize_splits(ds,tok)
        args=TrainingArguments(output_dir=str(out/'trainer'),eval_strategy='epoch',save_strategy='epoch',learning_rate=5e-5,per_device_train_batch_size=4,per_device_eval_batch_size=4,gradient_accumulation_steps=4,num_train_epochs=2,weight_decay=.01,max_grad_norm=1.,load_best_model_at_end=True,metric_for_best_model='f1',greater_is_better=True,logging_steps=50,report_to=[],seed=seed,save_total_limit=1)
        config.update(optimizer=args.optim.value,lr_scheduler=args.lr_scheduler_type.value,warmup_steps=args.warmup_steps,fp16=args.fp16,bf16=args.bf16,training_rows=len(tr))
        save_json(out/'config.json',config)
        trainer=Trainer(model=model,args=args,train_dataset=tokenized['train'],eval_dataset=tokenized['validation'],data_collator=DataCollatorWithPadding(tokenizer=tok),compute_metrics=upstream_module()._compute_metrics)
        trainer.train()
        trainer.save_model(str(out/'model'))
        tok.save_pretrained(str(out/'model'))
        save_json(out/'trainer_history.json',trainer.state.log_history)
        del trainer,model
        torch.cuda.empty_cache()
        evaluate(out/'model',out)
        save_json(out/'status.json',{'status':'complete','runtime_seconds':time.monotonic()-start})
    except Exception as exc:
        save_json(out/'status.json',{'status':'failed','runtime_seconds':time.monotonic()-start,'error':repr(exc)})
        raise

def source_sets():
    hard=pd.read_csv(DATA/'provguard_1000.csv',keep_default_na=False)
    hashes={}
    for source in ['dolly','oasst1']:
        selected=hard[hard.source.eq(source)].sort_values('rank').iloc[:250].copy()
        if len(selected)!=250:
            raise ValueError('Insufficient original approved source subset.')
        approved(selected)
        path=DATA/f'provguard_{source}_250.csv'
        if path.exists():
            old=pd.read_csv(path,keep_default_na=False)
            if not old.candidate_id.tolist()==selected.candidate_id.tolist() or not old.text.tolist()==selected.text.tolist():
                raise ValueError('Existing secondary source selection differs.')
        else:
            csv(selected,path)
        hashes[source]=sha(path)
    save_json(ROOT/'reports/source_transfer_selection.json',{'K_per_source':250,'seeds':SECONDARY_SEEDS,'selection':'highest-ranked original paired-semantic-approved ProvGuard rows within each source; equal K=250 chosen before source-isolated outcomes, not from test FPR','evaluation_n':{'dolly':68,'oasst1':140},'augmentation_sha256':hashes,'semantic_review_sha256':sha(DATA/'semantic_review.csv')})

def export_secondary(entries):
    rows=[]
    stats=[]
    for e in entries:
        if e['status']!='COMPLETED':
            continue
        out=Path(e['output_path'])
        c=json.loads((out/'config.json').read_text())
        test=rates(pd.read_csv(out/'test_predictions.csv'))
        hb=pd.read_csv(out/'hard_benign_predictions.csv',keep_default_na=False)
        h=rates(hb)
        row={'family':e['family'],'method':e['method'],'seed':e['seed'],'augmentation_size':c['augmentation_size'],'iid_f1':test['f1'],'iid_recall':test['recall'],'hard_benign_fpr':h['fpr'],'external_hb_fpr':rates(hb[hb.source_type.eq('real')])['fpr'],'obfuscated_recall':rates(pd.read_csv(out/'obfuscated_predictions.csv'))['recall'],'config_sha256':sha(out/'config.json')}
        for source in ['dolly','oasst1']:
            g=hb[hb.source_type.eq('real') & hb.source.str.startswith(source)]
            row[f'{source}_n']=len(g)
            row[f'{source}_fpr']=float((g.probability_score>=.5).mean())
        rows.append(row)
    frame=pd.DataFrame(rows)
    csv(frame,ROOT/'reports/SECONDARY_RESULTS.csv')
    lines=['# Secondary confirmations','', 'All required DistilBERT primary and ablation runs precede these confirmations. Models and failures are separate from primary hypothesis testing. No favorable seed is selected.','',md(pd.DataFrame(entries)[['family','method','seed','status','failure_reason']]),'']
    if len(frame):
        lines += [md(frame), '']
        for family,group in frame.groupby('family'):
            numeric=[c for c in ['iid_f1','iid_recall','hard_benign_fpr','external_hb_fpr','obfuscated_recall','dolly_fpr','oasst1_fpr'] if c in group]
            summary=group.groupby('method')[numeric].agg(['mean','std','count'])
            summary.to_csv(ROOT/f'reports/{family.upper()}_SUMMARY.csv')
    for family,methods in [('deberta',['provguard_1000']),('source',['provguard_dolly_250','provguard_oasst1_250'])]:
        for method in methods:
            matrices=[]
            used=[]
            for seed in SECONDARY_SEEDS:
                target=next((e for e in entries if e['family']==family and e['method']==method and e['seed']==seed and e['status']=='COMPLETED'),None)
                base=outpath('deberta','baseline',seed) if family=='deberta' else OUTPUT/'baseline'/f'seed_{seed}'
                if not target or not (base/'status.json').exists() or json.loads((base/'status.json').read_text()).get('status')!='complete':
                    continue
                a=pd.read_csv(base/'hard_benign_predictions.csv',keep_default_na=False)
                b=pd.read_csv(Path(target['output_path'])/'hard_benign_predictions.csv',keep_default_na=False)
                evaluation_source='all_external' if family=='deberta' else 'oasst1' if 'dolly' in method else 'dolly'
                mask=a.source_type.eq('real') if family=='deberta' else a.source_type.eq('real') & a.source.str.startswith(evaluation_source)
                mask_b=b.source_type.eq('real') if family=='deberta' else b.source_type.eq('real') & b.source.str.startswith(evaluation_source)
                p=align(a[mask],b[mask_b])
                delta=(p.probability_score_a>=.5).astype(int).to_numpy()-(p.probability_score_b>=.5).astype(int).to_numpy()
                matrices.append(delta)
                used.append(seed)
            if matrices:
                ci=crossed_bootstrap(matrices,seed=525252)
                stats.append({'family':family,'comparison':'baseline vs '+method,'evaluation_source':evaluation_source,'training_seed_n':len(used),'seeds':','.join(map(str,used)),'evaluation_n':len(matrices[0]),'absolute_fpr_reduction':float(np.mean(matrices)),'bootstrap_ci_lower':ci[0],'bootstrap_ci_upper':ci[1],'bootstrap_repetitions':10000,'analysis':'exploratory crossed training-seed/shared-prompt bootstrap; unadjusted pointwise CI'})
    stats=pd.DataFrame(stats)
    csv(stats,ROOT/'reports/SECONDARY_STATISTICS.csv')
    if len(stats):
        lines += [md(stats), '']
    lines += ['DeBERTa uses fixed max length 256, epochs 2, LR 5e-5, effective batch 16 (microbatch 4, accumulation 4), weight decay 0.01, max gradient norm 1.0, original-validation-F1 checkpoint selection and threshold 0.5 for both arms. These are secondary configuration choices, not method-specific tuning.','', 'Source-isolated models use equal K=250 from the already approved original hard set, with DistilBERT primary training settings and seeds 13,42,123. Dolly training is evaluated on 140 OASST external HB prompts; OASST training on 68 Dolly prompts. Small source subsets and three seeds limit inference.','', 'Additional independent random-control samples require new two-pass review from the full candidate pool; they are not synthesized from a hardness-biased reviewed subset.']
    (ROOT/'reports/SECONDARY_EXPERIMENT_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def run():
    import msvcrt
    global _secondary_lock_handle
    _secondary_lock_handle=(ROOT/'reports/secondary_gpu.lock').open('a+b')
    if _secondary_lock_handle.tell()==0:
        _secondary_lock_handle.write(b'0')
        _secondary_lock_handle.flush()
    _secondary_lock_handle.seek(0)
    try:
        msvcrt.locking(_secondary_lock_handle.fileno(),msvcrt.LK_NBLCK,1)
    except OSError:
        _secondary_lock_handle.close()
        print('Secondary confirmation controller already running; existing runs retained.')
        return
    # Keep this process awake through architecture, source and random controls.
    if sys.platform=='win32':
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(ctypes.c_uint32(0x80000001))
    environment()
    integrity()
    if not all(verify_completed(m,s) for m,s in PLAN):
        raise RuntimeError('Finish all thirty required DistilBERT runs before secondary confirmations.')
    source_sets()
    registry_path=ROOT/'reports/secondary_registry.json'
    previous=json.loads(registry_path.read_text()) if registry_path.exists() else {'experiments':[]}
    prior={(e['family'],e['method'],e['seed']):e for e in previous['experiments']}
    plan=[('deberta',m,s) for m in ['baseline','provguard_1000'] for s in SECONDARY_SEEDS]+[('source',f'provguard_{source}_250',s) for source in ['dolly','oasst1'] for s in SECONDARY_SEEDS]
    entries=[prior.get((f,m,s),{'family':f,'method':m,'seed':s,'status':'PENDING','failure_reason':None,'started_at':None,'completed_at':None,'output_path':str(outpath(f,m,s)),'attempts':[]}) for f,m,s in plan]
    def persist():
        save_json(registry_path,{'pid':os.getpid(),'experiments':entries})
        active=[e for e in entries if e['status']=='RUNNING']
        save_json(ROOT/'reports/secondary_controller_state.json',{'pid':os.getpid(),'status':'RUNNING','completed_run_n':sum(e['status']=='COMPLETED' for e in entries),'planned_run_n':len(entries),'current_run':f"{active[0]['family']}/{active[0]['method']}/{active[0]['seed']}" if active else None})
    logs=ROOT/'outputs/secondary_logs'
    logs.mkdir(parents=True,exist_ok=True)
    architecture_failed=False
    for e in entries:
        f,m,s=e['family'],e['method'],e['seed']
        augment=None if m=='baseline' else DATA/('provguard_1000.csv' if f=='deberta' else f'{m}.csv')
        if complete(f,m,s,augment):
            e.update(status='COMPLETED',failure_reason=None)
            persist()
            continue
        if f=='deberta' and architecture_failed:
            e.update(status='PENDING',failure_reason='Earlier architecture attempt failed; optional confirmation stopped without altering hyperparameters.')
            persist()
            continue
        if f=='deberta':
            cmd=[sys.executable,str(ROOT/'scripts/secondary_train.py'),'--method',m,'--seed',str(s)]
        else:
            out=outpath(f,m,s)
            if out.exists() and any(out.iterdir()):
                if (out/'model/model.safetensors').exists():
                    resume=True
                else:
                    e['attempts'].append({'archived_path':snapshot_incomplete(out),'reason':'Incomplete source attempt retained; safe optimizer restore unavailable.'})
                    resume=False
            else:
                resume=False
            cmd=[sys.executable,str(ROOT/'scripts/provguard.py'),'train','--method',m,'--seed',str(s),'--augment-train',str(augment)]+(['--resume'] if resume else [])
        if f=='deberta' and augment:
            cmd+=['--augment-train',str(augment)]
        log=logs/f'{f}_{m}_{s}_{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")}.log'
        e.update(status='RUNNING',started_at=datetime.now(timezone.utc).isoformat(),failure_reason=None,command=cmd,log_path=str(log))
        persist()
        with log.open('w',encoding='utf-8') as stream:
            result=subprocess.run(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        e['attempts'].append({'command':cmd,'log_path':str(log),'returncode':result.returncode})
        if result.returncode or not complete(f,m,s,augment):
            e.update(status='FAILED',failure_reason=f'Return code {result.returncode}; retained log: {log}')
            architecture_failed=f=='deberta'
        else:
            e.update(status='COMPLETED',completed_at=datetime.now(timezone.utc).isoformat(),model_sha256=sha(Path(e['output_path'])/'model/model.safetensors'))
        persist()
        export_secondary(entries)
    export_secondary(entries)
    save_json(ROOT/'reports/secondary_controller_state.json',{'pid':os.getpid(),'status':'COMPLETE' if all(e['status']=='COMPLETED' for e in entries) else 'PARTIAL','completed_run_n':sum(e['status']=='COMPLETED' for e in entries),'planned_run_n':len(entries)})
    from .robustness import run as random_robustness
    random_robustness()

