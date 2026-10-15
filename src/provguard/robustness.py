"""Additional random controls with separately versioned independent reviews."""
from __future__ import annotations
import json
import subprocess
import sys
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from .core import ROOT,DATA,OUTPUT,save_json,sha
from .study import PRIMARY,SEEDS,verify_completed,snapshot_incomplete
from .semantic import permutations,consensus,LABELS
from .final_analysis import csv,rates,paired_statistics,md

REVIEW_DIR=DATA/'random_robustness_reviews'
REVIEW_FILE=DATA/'semantic_review_robustness.csv'
SAMPLING_SEEDS=[2024,7777]

def prepare_reviews():
    if not all(verify_completed(m,s) for m in PRIMARY for s in SEEDS):
        raise RuntimeError('Sampling robustness follows completion of the twenty primary runs.')
    scored=pd.read_csv(DATA/'scored_candidates.csv',keep_default_na=False)
    hard=pd.read_csv(DATA/'provguard_1000.csv',keep_default_na=False)
    base=pd.read_csv(DATA/'semantic_review.csv',keep_default_na=False)
    known=set(base.candidate_id)
    wanted=set()
    orders={}
    for seed in SAMPLING_SEEDS:
        order=permutations(scored,seed)
        orders[seed]={f'{s}:{b}':ids for (s,b),ids in order.items()}
        for key,group in hard.groupby(['source','length_bin']):
            wanted.update(order[key][:len(group)+40])
    ids=sorted(wanted-known)
    queue=scored.set_index('candidate_id').loc[ids].reset_index()[['candidate_id','text','source']]
    REVIEW_DIR.mkdir(exist_ok=True)
    path=REVIEW_DIR/'additional_control_inputs.jsonl'
    if path.exists():
        existing=pd.read_json(path,lines=True,dtype={'candidate_id':str})
        if existing.candidate_id.tolist()!=queue.candidate_id.tolist() or existing.text.tolist()!=queue.text.tolist():
            raise ValueError('Additional control review queue changed; existing decisions must be preserved.')
    else:
        queue.to_json(path,orient='records',lines=True,force_ascii=False)
    save_json(ROOT/'reports/random_robustness_plan.json',{'sampling_seeds':SAMPLING_SEEDS,'training_seed':42,'source_length_quotas':{f'{s}:{b}':len(g) for (s,b),g in hard.groupby(['source','length_bin'])},'selection':'independent fixed full-pool permutations; first paired-semantic BENIGN candidates per stratum; no hardness condition and no forced disjointness','input_n':len(queue),'input_sha256':sha(path),'primary_review_sha256':sha(DATA/'semantic_review.csv'),'permutations_sha256':__import__('hashlib').sha256(json.dumps(orders,sort_keys=True).encode()).hexdigest(),'review_outputs':['pass_A.jsonl','pass_B.jsonl'],'primary_review_immutable':True})
    return queue

def build_controls():
    queue=prepare_reviews()
    parts=[]
    for pass_name in ['A','B']:
        path=REVIEW_DIR/f'pass_{pass_name}.jsonl'
        if not path.exists():
            raise RuntimeError(f'Additional control semantic pass {pass_name} is pending; no label-zero data was fabricated.')
        d=pd.read_json(path,lines=True,dtype={'candidate_id':str})
        if d.candidate_id.duplicated().any() or not set(d.decision).issubset(LABELS) or not set(d.candidate_id).issubset(set(queue.candidate_id)) or d.reason.map(lambda x:not isinstance(x,str) or not x.strip()).any():
            raise ValueError('Invalid additional semantic adjudication.')
        parts.append(d.rename(columns={'decision':f'review_{pass_name}','reason':f'reason_{pass_name}'}))
    paired=queue.merge(parts[0],on='candidate_id',validate='one_to_one').merge(parts[1],on='candidate_id',validate='one_to_one')
    paired['final_decision']=[consensus(a,b) for a,b in zip(paired.review_A,paired.review_B)]
    base=pd.read_csv(DATA/'semantic_review.csv',keep_default_na=False)
    base=base[['candidate_id','text','source','review_A','reason_A','review_B','reason_B','final_decision']]
    combined=pd.concat([base,paired],ignore_index=True)
    if combined.candidate_id.duplicated().any():
        raise ValueError('Additional adjudication duplicated original IDs.')
    scored=pd.read_csv(DATA/'scored_candidates.csv',keep_default_na=False)
    combined=scored.merge(combined,on=['candidate_id','text','source'],validate='one_to_one')
    hard=pd.read_csv(DATA/'provguard_1000.csv',keep_default_na=False)
    labels=combined.set_index('candidate_id').final_decision.to_dict()
    chosen={}
    for seed in SAMPLING_SEEDS:
        ids=[]
        for key,group in hard.groupby(['source','length_bin']):
            accepted=[]
            for cid in permutations(scored,seed)[key]:
                if cid not in labels:
                    raise RuntimeError(f'Additional random permutation requires missing paired review: {cid}')
                if labels[cid]=='BENIGN':
                    accepted.append(cid)
                if len(accepted)==len(group):
                    break
            if len(accepted)!=len(group):
                raise ValueError('Insufficient benign samples in an additional random stratum.')
            ids+=accepted
        chosen[seed]=ids
    csv(combined,REVIEW_FILE)
    from .leakage import LeakageIndex
    index=LeakageIndex.load()
    diagnostics=[]
    for seed,ids in chosen.items():
        frame=combined.set_index('candidate_id').loc[ids].reset_index()
        frame['label']=0
        checked,_=index.filter(frame)
        if len(checked)!=1000 or not checked.groupby(['source','length_bin']).size().equals(hard.groupby(['source','length_bin']).size()):
            raise ValueError('Additional random control leakage/strata check failed.')
        csv(checked,DATA/f'random_real_{seed}.csv')
        diagnostics.append({'sampling_seed':seed,'n':len(checked),'mean_attack_probability':checked.mean_attack_probability.mean(),'median_attack_probability':checked.mean_attack_probability.median(),'overlap_with_provguard':len(set(ids)&set(hard.candidate_id)),'augmentation_sha256':sha(DATA/f'random_real_{seed}.csv')})
    save_json(ROOT/'reports/random_robustness_selection_audit.json',{'diagnostics':diagnostics,'review_sha256':sha(REVIEW_FILE),'primary_review_sha256':sha(DATA/'semantic_review.csv'),'extra_paired_review_n':len(paired),'extra_decisions':paired.final_decision.value_counts().to_dict(),'process':'two independent full-text automated semantic passes, scores withheld; primary reviews preserved separately','decision_sha256':{p.name:sha(p) for p in REVIEW_DIR.glob('pass_*.jsonl')}})

def run():
    try:
        build_controls()
    except RuntimeError as exc:
        save_json(ROOT/'reports/random_robustness_state.json',{'status':'PENDING_REVIEWS','reason':str(exc),'review_input':str(REVIEW_DIR/'additional_control_inputs.jsonl')})
        return
    registry=ROOT/'reports/random_robustness_registry.json'
    old=json.loads(registry.read_text()) if registry.exists() else {'experiments':[]}
    byseed={r['sampling_seed']:r for r in old['experiments']}
    entries=[]
    for sampling_seed in SAMPLING_SEEDS:
        method=f'random_real_{sampling_seed}'
        row=byseed.get(sampling_seed,{'sampling_seed':sampling_seed,'method':method,'seed':42,'status':'PENDING','failure_reason':None,'attempts':[]})
        entries.append(row)
        if verify_completed(method,42):
            row['status']='COMPLETED'
            continue
        out=OUTPUT/method/'seed_42'
        resume=False
        if out.exists() and any(out.iterdir()):
            if (out/'model/model.safetensors').exists():
                resume=True
            else:
                row['attempts'].append({'archived_path':snapshot_incomplete(out),'reason':'Incomplete attempt retained; no safe optimizer resume in current environment.'})
        cmd=[sys.executable,str(ROOT/'scripts/provguard.py'),'train','--method',method,'--seed','42','--augment-train',str(DATA/f'{method}.csv'),'--semantic-review',str(REVIEW_FILE)]+(['--resume'] if resume else [])
        log=ROOT/f'outputs/secondary_logs/{method}_42_{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")}.log'
        log.parent.mkdir(exist_ok=True,parents=True)
        row.update(status='RUNNING',command=cmd,log_path=str(log),started_at=datetime.now(timezone.utc).isoformat())
        save_json(registry,{'experiments':entries})
        with log.open('w',encoding='utf-8') as stream:
            result=subprocess.run(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        row['attempts'].append({'command':cmd,'log_path':str(log),'returncode':result.returncode})
        row.update(status='COMPLETED' if not result.returncode and verify_completed(method,42) else 'FAILED',failure_reason=None if not result.returncode else f'Return code {result.returncode}; inspect {log}')
        save_json(registry,{'experiments':entries})
    rows=[]
    stats=[]
    for method,sampling_seed in [('random_real',42),('random_real_2024',2024),('random_real_7777',7777)]:
        if not verify_completed(method,42):
            continue
        out=OUTPUT/method/'seed_42'
        test=rates(pd.read_csv(out/'test_predictions.csv'))
        h=pd.read_csv(out/'hard_benign_predictions.csv',keep_default_na=False)
        rows.append({'method':method,'sampling_seed':sampling_seed,'training_seed':42,'iid_f1':test['f1'],'iid_recall':test['recall'],'hard_benign_fpr':rates(h)['fpr'],'external_hb_fpr':rates(h[h.source_type.eq('real')])['fpr'],'obfuscated_recall':rates(pd.read_csv(out/'obfuscated_predictions.csv'))['recall']})
        pg=pd.read_csv(OUTPUT/'provguard_1000/seed_42/hard_benign_predictions.csv',keep_default_na=False)
        for subset in ['usable_hard_benign','external_hard_benign']:
            a=h if subset.startswith('usable') else h[h.source_type.eq('real')]
            b=pg if subset.startswith('usable') else pg[pg.source_type.eq('real')]
            record,_=paired_statistics(a,b,seed=625252)
            stats.append({'method':method,'sampling_seed':sampling_seed,'training_seed':42,'subset':subset,**record})
    frame=pd.DataFrame(rows)
    csv(frame,ROOT/'reports/RANDOM_CONTROL_ROBUSTNESS.csv')
    csv(pd.DataFrame(stats),ROOT/'reports/RANDOM_CONTROL_ROBUSTNESS_STATISTICS.csv')
    (ROOT/'reports/RANDOM_CONTROL_ROBUSTNESS.md').write_text('# Additional random-control samples\n\nFixed sampling seeds 42 (primary), 2024 and 7777; training seed 42 for every control. These exploratory comparisons check sampling sensitivity and do not replace five-seed primary inference. Controls are independently sampled from the full corpus without hardness filtering, matched on source/length strata, and accepted only after the same two-pass semantic criterion.\n\n'+md(frame)+'\n\n'+md(pd.DataFrame(stats))+'\n',encoding='utf-8')
    save_json(ROOT/'reports/random_robustness_state.json',{'status':'COMPLETE' if len(rows)==3 else 'PARTIAL','additional_model_n':len(rows)-1})

