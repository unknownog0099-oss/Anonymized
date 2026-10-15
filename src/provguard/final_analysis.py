"""Canonical measured exports; final claims require all thirty controlled runs."""
from __future__ import annotations
import json
import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from .core import ROOT, DATA, OUTPUT, SETS, BENCH, read_texts, sha, save_json
from .study import SEEDS, METHODS, PRIMARY, PLAN, augmentation, verify_completed

REPORT = ROOT/'reports'
METRICS = ['iid_f1','iid_precision','iid_recall','iid_benign_fpr','roc_auc','hard_benign_fpr','external_hb_fpr','curated_hb_fpr','obfuscated_recall','domain_ood_f1','structural_ood_f1','structural_ood_fpr']
DISPLAY = ['iid_f1','hard_benign_fpr','external_hb_fpr','iid_recall','obfuscated_recall','domain_ood_f1','structural_ood_fpr']
STAT_COLUMNS=['comparison','subset','seed','n','training_seed_n','FPR_A','FPR_B','delta_FPR','fp_fixed','new_fp','discordant_n','mcnemar_statistic','p_value','p_holm','bootstrap_ci_lower','bootstrap_ci_upper','bootstrap_repetitions','bootstrap_seed','analysis']
ERROR_COLUMNS=['seed','subset','row_id','text','source_type','baseline_prediction','provguard_prediction','true_label','baseline_probability','provguard_probability','transition_type','error_category','categorization_method']

def csv(frame,path):
    path=Path(path)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    frame.to_csv(tmp,index=False)
    tmp.replace(path)

def predictions(method,seed,subset):
    return pd.read_csv(OUTPUT/method/f'seed_{seed}'/f'{subset}_predictions.csv',keep_default_na=False)

def rates(frame):
    y=frame.true_label.astype(int).to_numpy()
    p=frame.probability_score.to_numpy(dtype=float)
    pred=p>=.5
    benign=y==0
    attacks=y==1
    return {'n':len(y),'f1':f1_score(y,pred,zero_division=0),'precision':precision_score(y,pred,zero_division=0),'recall':float(pred[attacks].mean()) if attacks.any() else None,'fpr':float(pred[benign].mean()) if benign.any() else None,'roc_auc':roc_auc_score(y,p) if benign.any() and attacks.any() else None}

def collect():
    rows=[]
    for method,seed in PLAN:
        if not verify_completed(method,seed):
            continue
        out=OUTPUT/method/f'seed_{seed}'
        config=json.loads((out/'config.json').read_text())
        status=json.loads((out/'status.json').read_text())
        test=rates(predictions(method,seed,'test'))
        hb=predictions(method,seed,'hard_benign')
        h=rates(hb)
        external=rates(hb[hb.source_type.eq('real')])
        curated=rates(hb[hb.source_type.eq('curated')])
        structural=rates(predictions(method,seed,'structural_ood'))
        row={'method':method,'seed':seed,'augmentation_size':config['augmentation_size'],'augmentation_file':config.get('augmentation_file'),'augmentation_sha256':config.get('augmentation_sha256'),'iid_f1':test['f1'],'iid_precision':test['precision'],'iid_recall':test['recall'],'iid_benign_fpr':test['fpr'],'iid_fpr':test['fpr'],'roc_auc':test['roc_auc'],'hard_benign_n':h['n'],'hard_benign_fpr':h['fpr'],'external_hb_n':external['n'],'external_hb_fpr':external['fpr'],'curated_hb_n':curated['n'],'curated_hb_fpr':curated['fpr'],'obfuscated_recall':rates(predictions(method,seed,'obfuscated'))['recall'],'domain_ood_f1':rates(predictions(method,seed,'domain_ood'))['f1'],'structural_ood_f1':structural['f1'],'structural_ood_fpr':structural['fpr'],'balanced_subtype_f1':rates(predictions(method,seed,'balanced_subtype'))['f1'],'threshold':.5,'runtime_seconds':status.get('runtime_seconds'),'runtime_note':status.get('runtime_note','training plus evaluation'),'status':'COMPLETED','config_file':str(out/'config.json'),'config_sha256':sha(out/'config.json'),'evaluation_sha256':sha(out/'evaluation.json')}
        rows.append(row)
    frame=pd.DataFrame(rows)
    csv(frame,ROOT/'results_master.csv')
    return frame

def summary(frame,methods):
    rows=[]
    for method in methods:
        g=frame[frame.method.eq(method)]
        complete=set(g.seed.astype(int))==set(SEEDS)
        row={'method':method,'K':int(method.split('_')[-1]) if method.startswith('provguard_') else None,'augmentation_size':int(g.augmentation_size.iloc[0]) if len(g) else None,'completed_seeds':','.join(str(s) for s in sorted(g.seed.astype(int))), 'seed_n':len(g),'status':'COMPLETE' if complete else 'INCOMPLETE'}
        for metric in METRICS:
            row[metric+'_mean']=float(g[metric].mean()) if len(g) else None
            row[metric+'_sd']=float(g[metric].std(ddof=1)) if len(g)>1 else None
        rows.append(row)
    return pd.DataFrame(rows)

def align(left,right):
    if left.eval_id.duplicated().any() or right.eval_id.duplicated().any() or set(left.eval_id)!=set(right.eval_id):
        raise ValueError('Paired prediction IDs differ or repeat.')
    p=left[['eval_id','text','true_label','probability_score']].merge(right[['eval_id','text','true_label','probability_score']],on='eval_id',suffixes=('_a','_b'),validate='one_to_one').sort_values('eval_id').reset_index(drop=True)
    if not p.text_a.eq(p.text_b).all() or not p.true_label_a.eq(p.true_label_b).all():
        raise ValueError('Paired prediction text/labels differ.')
    return p

def crossed_bootstrap(delta,repetitions=10000,seed=42):
    """Resample training seeds and shared row IDs independently, never pool S*N."""
    delta=np.asarray(delta,dtype=float)
    if delta.ndim!=2 or not delta.shape[0] or not delta.shape[1]:
        raise ValueError('Need nonempty seed by row deltas.')
    s,n=delta.shape
    patterns,counts=np.unique(delta.T,axis=0,return_counts=True)
    rng=np.random.default_rng(seed)
    row_draws=rng.multinomial(n,counts/n,size=repetitions)
    seed_draws=rng.multinomial(s,np.full(s,1/s),size=repetitions)
    boot=((row_draws@patterns)*seed_draws).sum(axis=1)/(n*s)
    return np.quantile(boot,[.025,.975])

def paired_statistics(left,right,repetitions=10000,seed=42):
    p=align(left,right)
    if not p.true_label_a.eq(0).all() or p.empty:
        raise ValueError('FPR pairing requires benign rows.')
    a=p.probability_score_a.to_numpy()>=.5
    b=p.probability_score_b.to_numpy()>=.5
    fixed=int((a & ~b).sum())
    new=int((~a & b).sum())
    delta=a.astype(int)-b.astype(int)
    ci=crossed_bootstrap(delta[None,:],repetitions,seed)
    return {'n':len(p),'training_seed_n':1,'FPR_A':float(a.mean()),'FPR_B':float(b.mean()),'delta_FPR':float(delta.mean()),'fp_fixed':fixed,'new_fp':new,'discordant_n':fixed+new,'mcnemar_statistic':(fixed-new)**2/(fixed+new) if fixed+new else 0.,'p_value':float(binomtest(min(fixed,new),fixed+new,.5).pvalue) if fixed+new else 1.,'bootstrap_ci_lower':float(ci[0]),'bootstrap_ci_upper':float(ci[1]),'bootstrap_repetitions':repetitions,'bootstrap_seed':seed,'analysis':'paired prompt bootstrap and exact binomial McNemar; delta=A minus ProvGuard'},delta

def statistics(frame):
    rows=[]
    have=set(zip(frame.method,frame.seed.astype(int)))
    for comparator in ['baseline','random_real','curated']:
        for subset in ['usable_hard_benign','external_hard_benign']:
            matrices=[]
            fprs=[]
            used=[]
            common_ids=None
            for seed in SEEDS:
                if (comparator,seed) not in have or ('provguard_1000',seed) not in have:
                    continue
                a=predictions(comparator,seed,'hard_benign')
                b=predictions('provguard_1000',seed,'hard_benign')
                if subset.startswith('external'):
                    a=a[a.source_type.eq('real')]
                    b=b[b.source_type.eq('real')]
                ids=sorted(a.eval_id)
                if common_ids is not None and ids!=common_ids:
                    raise ValueError('Evaluation rows differ across seeds.')
                common_ids=ids
                record,delta=paired_statistics(a,b,seed=424200+seed)
                record.update(comparison=f'{comparator} vs provguard_1000',subset=subset,seed=seed)
                rows.append(record)
                matrices.append(delta)
                fprs.append([record['FPR_A'],record['FPR_B']])
                used.append(seed)
            if matrices:
                ci=crossed_bootstrap(matrices,seed=424242)
                means=np.mean(fprs,axis=0)
                rows.append({'comparison':f'{comparator} vs provguard_1000','subset':subset,'seed':'AGGREGATE','n':len(common_ids),'training_seed_n':len(used),'FPR_A':means[0],'FPR_B':means[1],'delta_FPR':means[0]-means[1],'bootstrap_ci_lower':ci[0],'bootstrap_ci_upper':ci[1],'bootstrap_repetitions':10000,'bootstrap_seed':424242,'analysis':'crossed bootstrap resamples paired training seeds and shared prompt IDs; no pooled McNemar p-value; '+('all five seeds' if len(used)==5 else 'INCOMPLETE seeds')})
    result=pd.DataFrame(rows,columns=STAT_COLUMNS)
    if len(result):
        idx=result.index[result.seed.ne('AGGREGATE')]
        ordered=sorted(idx,key=lambda i:result.loc[i,'p_value'])
        adjusted=0.
        for rank,i in enumerate(ordered):
            adjusted=max(adjusted,min(1.,result.loc[i,'p_value']*(len(ordered)-rank)))
            result.loc[i,'p_holm']=adjusted
    csv(result,REPORT/'STATISTICAL_RESULTS.csv')
    save_json(REPORT/'statistical_protocol.json',{'delta_sign':'comparator FPR minus ProvGuard FPR; positive favors ProvGuard','resamples':10000,'per_seed_bootstrap_seed':'424200 + training seed','aggregate_bootstrap_seed':424242,'aggregate_unit':'training seed and shared evaluation row; crossed resampling preserves pairing and row dependence across seeds','multiple_testing':'Holm correction across all available per-seed exact McNemar tests (30 at completion); CIs are pointwise 95%, not simultaneous','aggregate_p_value':None,'scope':'Predefined comparisons and usable/external HB subsets; stress-set inference is conditional on public sample coverage.'})
    return result

def transition(y,a,b):
    if y==0:
        return 'FP_FIXED' if a==1 and b==0 else 'NEW_FP' if a==0 and b==1 else 'FP_REMAINS' if a==b==1 else 'UNCHANGED_CORRECT'
    return 'FN_FIXED' if a==0 and b==1 else 'NEW_FN' if a==1 and b==0 else 'FN_REMAINS' if a==b==0 else 'UNCHANGED_CORRECT'

def category(text,source=None):
    """Exploratory lexical categories only, never used for benign adjudication."""
    released={'hard_benign_security_education':'security education','hard_benign_attack_analysis':'quoted attack language','hard_benign_imperative_benign':'benign imperative','hard_benign_policy_meta':'system-prompt discussion','hard_benign_redteam_framing':'defensive red-team/security','hard_benign_guardrail_design':'defensive red-team/security','hard_benign_technical_documentation':'technical documentation','hard_benign_fiction_roleplay_benign':'other'}
    if source in released:
        return released[source]
    t=text.lower()
    if re.search(r'quot(?:e|ed|ing)|example.{0,50}(ignore|jailbreak)|phrase.{0,50}(ignore|attack)',t):
        return 'quoted attack language'
    if re.search(r'system prompt|developer (message|instruction)|instruction hierarchy',t):
        return 'system-prompt discussion'
    if re.search(r'defen[cs]|red[- ]team|penetration test|incident response|threat model',t):
        return 'defensive red-team/security'
    if re.search(r'prompt injection|jailbreak|cybersecurity|security|malware|phishing',t):
        return 'security education'
    if re.search(r'documentation|```|\b(?:api|function|compiler|python|javascript|configuration)\b',t):
        return 'technical documentation'
    if re.search(r'\b(ignore|follow|must|do not|don.t|please|ensure)\b',t):
        return 'benign imperative'
    if re.search(r'^(write|give|explain|describe|summarize|list|create|generate|how)',t.strip()):
        return 'general instruction'
    return 'other'

def error_analysis(frame):
    have=set(zip(frame.method,frame.seed.astype(int)))
    pieces=[]
    counts=[]
    for seed in SEEDS:
        if ('baseline',seed) in have and ('provguard_1000',seed) in have:
            for subset in ['test','hard_benign','balanced_subtype','obfuscated','domain_ood','structural_ood']:
                b=predictions('baseline',seed,subset)
                p=predictions('provguard_1000',seed,subset)
                paired=align(b,p)
                original=b.set_index('eval_id')
                a=(paired.probability_score_a>=.5).astype(int)
                z=(paired.probability_score_b>=.5).astype(int)
                result=pd.DataFrame({'seed':seed,'subset':subset,'row_id':paired.eval_id,'text':paired.text_a,'source_type':paired.eval_id.map(original.source_type) if 'source_type' in original else subset,'baseline_prediction':a,'provguard_prediction':z,'true_label':paired.true_label_a,'baseline_probability':paired.probability_score_a,'provguard_probability':paired.probability_score_b})
                result['transition_type']=[transition(y,x,v) for y,x,v in zip(result.true_label,a,z)]
                origins=paired.eval_id.map(original.source) if 'source' in original else [None]*len(result)
                result['error_category']=[category(t,s) for t,s in zip(result.text,origins)]
                result['categorization_method']='released curated source taxonomy where available, otherwise ordered lexical proxy; automated, exploratory'
                pieces.append(result)
        for method in ['baseline','random_real','provguard_1000']:
            if (method,seed) not in have:
                continue
            h=predictions(method,seed,'hard_benign')
            h['error_category']=[category(t,s) for t,s in zip(h.text,h.source)]
            for cat,g in h.groupby('error_category'):
                counts.append({'method':method,'seed':seed,'error_category':cat,'n':len(g),'false_positive_n':int((g.probability_score>=.5).sum()),'fpr':float((g.probability_score>=.5).mean()),'categorization_method':'released curated source taxonomy where available, otherwise ordered lexical proxy'})
    errors=pd.concat(pieces,ignore_index=True) if pieces else pd.DataFrame(columns=ERROR_COLUMNS)
    csv(errors[ERROR_COLUMNS],REPORT/'ERROR_TRANSITIONS.csv')
    categories=pd.DataFrame(counts,columns=['method','seed','error_category','n','false_positive_n','fpr','categorization_method'])
    csv(categories,REPORT/'ERROR_CATEGORIES.csv')
    if len(errors):
        csv(errors.groupby(['seed','subset','transition_type']).size().reset_index(name='n'),REPORT/'ERROR_TRANSITION_COUNTS.csv')
    save_json(REPORT/'error_category_protocol.json',{'method':'released curated source taxonomy mapping, then ordered lexical proxies for real/unmapped prompts; automated and exploratory, not human annotation','order':['quoted attack language','system-prompt discussion','defensive red-team/security','security education','technical documentation','benign imperative','general instruction','other'],'source_sha256':sha(Path(__file__)),'limitation':'Categories overlap semantically; first matching lexical rule wins. Released curated source labels supply eight known prompt families; real prompts use approximate lexical proxies. These descriptive categories do not affect training, adjudication, or hypothesis tests.'})
    return categories

def comparisons(frame):
    rows=[]
    for comparator in ['baseline','random_real','curated']:
        a=frame[frame.method.eq(comparator)].set_index('seed')
        b=frame[frame.method.eq('provguard_1000')].set_index('seed')
        common=sorted(set(a.index)&set(b.index))
        if not common:
            continue
        x=a.loc[common,METRICS].mean()
        y=b.loc[common,METRICS].mean()
        row={'comparison':f'{comparator} vs provguard_1000','matched_seed_n':len(common),'matched_seeds':','.join(map(str,common)),'status':'COMPLETE' if set(common)==set(SEEDS) else 'INCOMPLETE'}
        for metric in ['hard_benign_fpr','external_hb_fpr']:
            row[metric+'_absolute_reduction']=x[metric]-y[metric]
            row[metric+'_relative_reduction']=(x[metric]-y[metric])/x[metric] if x[metric] else None
        for metric in ['iid_f1','iid_recall','obfuscated_recall','domain_ood_f1','structural_ood_fpr']:
            row[metric+'_provguard_minus_comparator']=y[metric]-x[metric]
        rows.append(row)
    result=pd.DataFrame(rows)
    csv(result,REPORT/'COMPARISON_RESULTS.csv')
    return result

def curated_hardness():
    from .study import running_training
    if running_training() or not (REPORT/'candidate_selection_audit.json').exists():
        return
    path=DATA/'curated_hn.csv'
    scored_path=DATA/'curated_scored.csv'
    provenance=REPORT/'curated_selector_score_provenance.json'
    if scored_path.exists() and provenance.exists():
        previous=json.loads(provenance.read_text())
        if previous['augmentation_sha256']!=sha(path) or previous['scored_sha256']!=sha(scored_path):
            raise ValueError('Curated hardness provenance changed.')
        scored=pd.read_csv(scored_path,keep_default_na=False)
    else:
        from .model import probabilities
        scored=pd.read_csv(path,keep_default_na=False)
        for seed in [13,42,123]:
            if not verify_completed('baseline',seed):
                raise ValueError('Curated diagnostic requires completed fixed selectors.')
            scored[f'score_seed{seed}']=probabilities(OUTPUT/'baseline'/f'seed_{seed}'/'model',scored.text.tolist())
        scored['mean_attack_probability']=scored[[f'score_seed{s}' for s in [13,42,123]]].mean(axis=1)
        csv(scored,scored_path)
        save_json(provenance,{'augmentation_sha256':sha(path),'scored_sha256':sha(scored_path),'selector_seeds':[13,42,123],'selector_model_sha256':{str(s):sha(OUTPUT/'baseline'/f'seed_{s}'/'model/model.safetensors') for s in [13,42,123]},'purpose':'post-selection hardness diagnostic only; does not select or alter curated augmentation'})
    x=scored.mean_attack_probability
    row={'method':'curated','n':len(scored),'min':x.min(),'q1':x.quantile(.25),'median':x.median(),'mean':x.mean(),'q3':x.quantile(.75),'max':x.max(),**{f'n_above_{t}':int((x>t).sum()) for t in [.5,.25,.1,.05,.01]}}
    table=pd.read_csv(REPORT/'HARDNESS_ANALYSIS.csv')
    table=table[~table.method.eq('curated')]
    csv(pd.concat([table,pd.DataFrame([row])],ignore_index=True),REPORT/'HARDNESS_ANALYSIS.csv')

def source_stratified(frame):
    rows=[]
    for _,run in frame.iterrows():
        h=predictions(run['method'],int(run.seed),'hard_benign')
        for source in ['dolly','oasst1']:
            g=h[h.source_type.eq('real') & h.source.str.startswith(source)]
            rows.append({'method':run['method'],'seed':int(run.seed),'evaluation_source':source,'n':len(g),'false_positive_n':int((g.probability_score>=.5).sum()),'fpr':float((g.probability_score>=.5).mean()),'analysis':'source-stratified public external HB evaluation; mixed-source real augmentation, not source-isolated transfer'})
    result=pd.DataFrame(rows)
    csv(result,REPORT/'SOURCE_STRATIFIED_RESULTS.csv')
    grouped=result.groupby(['method','evaluation_source']).agg(seed_n=('seed','count'),n=('n','first'),fpr_mean=('fpr','mean'),fpr_sd=('fpr','std')).reset_index()
    csv(grouped,REPORT/'SOURCE_STRATIFIED_SUMMARY.csv')
    return grouped

def markdown_table(frame,metrics=DISPLAY):
    lines=['| Method | Seeds | '+ ' | '.join(metrics)+' |','|---|---:|'+ '|'.join(['---:']*len(metrics))+'|']
    for _,r in frame.iterrows():
        vals=[]
        for m in metrics:
            mean,sd=r[m+'_mean'],r[m+'_sd']
            vals.append('unavailable' if pd.isna(mean) else f'{mean:.4f} ± {sd:.4f}' if pd.notna(sd) else f'{mean:.4f} (one seed)')
        lines.append(f"| {r['method']} | {r['seed_n']}/5 | "+' | '.join(vals)+' |')
    return '\n'.join(lines)

def md(frame,precision=4):
    # Keep exports dependency-free; escape prompt text if used in a table.
    def value(x):
        if pd.isna(x):
            return 'unavailable'
        if isinstance(x,(float,np.floating)):
            return f'{x:.{precision}f}'
        return str(x).replace('|','\\|').replace('\n',' ')
    lines=['| '+' | '.join(map(str,frame.columns))+' |','|'+'|'.join(['---']*len(frame.columns))+'|']
    lines += ['| '+' | '.join(value(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)]
    return '\n'.join(lines)

def report(frame,primary,ablation,stats,comparison,final):
    done=set(zip(frame.method,frame.seed.astype(int)))
    complete=all(key in done for key in PLAN)
    verdict={'H1':'INCONCLUSIVE','H2':'INCONCLUSIVE','H3':'INCONCLUSIVE','outcome':'INCONCLUSIVE','practical_success':None,'required_runs_complete':complete,'completed_run_n':len(done)}
    if len(stats) and all((m,s) in done for m in PRIMARY for s in SEEDS):
        verdict['endpoint_verdicts']={}
        for h,m in [('H1','baseline'),('H2','random_real')]:
            endpoints={}
            for subset in ['usable_hard_benign','external_hard_benign']:
                r=stats[stats.comparison.eq(f'{m} vs provguard_1000') & stats.subset.eq(subset) & stats.seed.eq('AGGREGATE')].iloc[0]
                endpoints[subset]={'verdict':'SUPPORTED' if r.bootstrap_ci_lower>0 else 'NOT SUPPORTED' if r.bootstrap_ci_upper<=0 else 'INCONCLUSIVE','absolute_fpr_reduction':float(r.delta_FPR),'ci_lower':float(r.bootstrap_ci_lower),'ci_upper':float(r.bootstrap_ci_upper)}
            verdict['endpoint_verdicts'][h]=endpoints
            labels=[v['verdict'] for v in endpoints.values()]
            verdict[h]='SUPPORTED' if all(v=='SUPPORTED' for v in labels) else 'NOT SUPPORTED' if all(v=='NOT SUPPORTED' for v in labels) else 'INCONCLUSIVE'
        c=comparison[comparison.comparison.eq('baseline vs provguard_1000')].iloc[0]
        passed=bool(c.external_hb_fpr_absolute_reduction>=.10 and c.iid_f1_provguard_minus_comparator>=-.01 and c.iid_recall_provguard_minus_comparator>=-.02)
        verdict['practical_success']=passed
        b=frame[frame.method.eq('baseline')].set_index('seed').loc[SEEDS]
        p=frame[frame.method.eq('provguard_1000')].set_index('seed').loc[SEEDS]
        rng=np.random.default_rng(424243)
        draw=rng.integers(0,5,size=(10000,5))
        uncertainty={}
        for metric in ['iid_f1','iid_recall','obfuscated_recall']:
            delta=(p[metric]-b[metric]).to_numpy()
            ci=np.quantile(delta[draw].mean(axis=1),[.025,.975])
            uncertainty[metric]={'mean_delta':float(delta.mean()),'seed_bootstrap_ci_lower':float(ci[0]),'seed_bootstrap_ci_upper':float(ci[1])}
        verdict['attack_preservation']=uncertainty
        f, r = uncertainty['iid_f1'],uncertainty['iid_recall']
        verdict['H3']='SUPPORTED' if f['seed_bootstrap_ci_lower']>=-.01 and r['seed_bootstrap_ci_lower']>=-.02 else 'NOT SUPPORTED' if f['seed_bootstrap_ci_upper']<-.01 or r['seed_bootstrap_ci_upper']<-.02 else 'INCONCLUSIVE'
        obfuscated=uncertainty['obfuscated_recall']
        if obfuscated['seed_bootstrap_ci_upper']<0:
            verdict['H3']='NOT SUPPORTED'
        elif obfuscated['mean_delta']<0 and verdict['H3']=='SUPPORTED':
            verdict['H3']='INCONCLUSIVE'
        if verdict['H1']==verdict['H2']==verdict['H3']=='SUPPORTED':
            verdict['outcome']='STRONG SUCCESS' if passed else 'MODERATE SUCCESS'
        elif verdict['H2']=='NOT SUPPORTED':
            verdict['outcome']='NO TARGETED-SELECTION ADVANTAGE'
        elif verdict['H3']=='NOT SUPPORTED':
            verdict['outcome']='MIXED RESULT' if verdict['H1']=='SUPPORTED' else 'FAILED / NOT SUPPORTED'
    save_json(REPORT/'hypothesis_verdicts.json',verdict)
    lines=['# ProvGuard experiment report','',f"Required runs completed: {len(done)}/30. Status: {'COMPLETE' if complete else 'INCOMPLETE'}. Generated from verified prediction files at threshold 0.5. No paper was written.",'','## 1. Completion matrix','','| Method | '+' | '.join(map(str,SEEDS))+' |','|---|'+'|'.join(['---']*5)+'|']
    for method in METHODS:
        lines.append('| '+method+' | '+' | '.join('COMPLETED' if (method,s) in done else 'PENDING / see registry' for s in SEEDS)+' |')
    lines+=['','## 2. Main result table','',markdown_table(primary),'','Means and sample SDs use every completed seed. Rows below five seeds are incomplete and cannot support final controlled claims. Hard-benign coverage: 808 usable public texts (208 external, 600 curated); 664 withheld rows excluded.','', '## 3. Raw per-seed table','',md(frame[['method','seed','augmentation_size',*DISPLAY]]),'','## 4. Ablation table','',markdown_table(ablation,['iid_f1','iid_recall','hard_benign_fpr','external_hb_fpr','obfuscated_recall']),'','## 5. Hardness validation','']
    if (REPORT/'HARDNESS_ANALYSIS.csv').exists() and (REPORT/'candidate_selection_audit.json').exists():
        hardness=pd.read_csv(REPORT/'HARDNESS_ANALYSIS.csv')
        audit=json.loads((REPORT/'candidate_selection_audit.json').read_text())
        lines+=[md(hardness,6),'',f"Selection uses baseline seeds 13,42,123, descending mean score and two independent full-text semantic passes. Rank prefix checked through {audit['rank_complete_through']}. Random seed 42 draws fixed permutations independently of hardness within source/length strata. Overlap: {audit['random_overlap_n']}. Nested sizes: 250,500,1000. Review is automated semantic adjudication, not human annotation."]
    else:
        lines+=['Pending complete independent adjudication and corrected set construction. Older easy sets are excluded by the training guards.']
    lines+=['','## 6. Statistical results','',md(stats,6) if len(stats) else 'No valid paired treatment outputs yet.','', 'Positive delta means comparator FPR minus ProvGuard FPR. Exact McNemar tests are per seed with Holm correction. Aggregate 10,000-resample crossed bootstrap resamples training seeds and shared prompts; predictions across seeds are never pooled as independent examples. CIs are pointwise 95%.','', '## 7. Error analysis','', 'ERROR_TRANSITIONS.csv includes all aligned rows, including remaining errors and unchanged correct rows. ERROR_CATEGORIES.csv maps released curated source taxonomy to categories and uses explicit ordered lexical proxies for real prompts. Categorization is exploratory, automated, and separate from semantic validity adjudication.','', '## 8. H1/H2/H3 verdicts','',f"H1: {verdict['H1']}. H2 (primary novelty test): {verdict['H2']}. H3: {verdict['H3']}.",f"Final outcome: {verdict['outcome']}. Practical project criterion satisfied: {verdict['practical_success']}.",'','H1/H2 support conservatively requires positive aggregate lower CI bounds for both usable and external HB; mixed endpoint evidence is inconclusive, with each endpoint verdict saved separately. H3 compares paired seed-bootstrap IID F1/recall changes against predefined loss bounds 0.01/0.02. A confirmed obfuscated-recall decline vetoes preservation; a negative point estimate with uncertain sign is inconclusive. No obfuscated noninferiority margin was predefined, so no formal margin claim is made. Five training seeds give limited uncertainty resolution.','', '## 9. Security-usability interpretation','',md(comparison,6) if len(comparison) else 'Pending treatment results.','', 'ProvGuard versus equal-size matched Random-Real is the primary selection comparison. Curated-HN is secondary and has a different size. Public stress-set FPR is not a deployment prevalence estimate.','', '## 10. Best configuration','', 'Primary configuration remains preregistered K=1000, DistilBERT, max length 256, epochs 2, effective batch 16, LR 5e-5, weight decay 0.01, max gradient norm 1.0. Original validation F1 selects the checkpoint. No preferred K is chosen from test FPR alone.','', '## 11. conference readiness','', 'Core experiment matrix and outputs complete; review the measured hypothesis verdicts and fallible automated adjudication before paper writing.' if complete else 'Research is incomplete. Missing experiments and analyses prevent publication claims. Resume from experiment_registry.json; completed models are preserved.','', 'Architecture transfer, source transfer and additional random samples are secondary. No such result is claimed without a separate measured output. External HB source labels identify 68 Dolly and 140 OASST texts. SOURCE_STRATIFIED_RESULTS.csv reports mixed-source training performance separately on these populations; source-isolated training is a distinct secondary confirmation.','', '## 12. Output paths and reproducibility','', '`results_master.csv`; `reports/{PRIMARY_RESULTS,ABLATION_RESULTS,STATISTICAL_RESULTS,HARDNESS_ANALYSIS,ERROR_TRANSITIONS,ERROR_CATEGORIES,COMPARISON_RESULTS}.csv`; `reports/experiment_registry.json`; `reports/environment_report.json`; `reports/canonical_leakage_audit.json`; `reports/semantic_adjudication_audit.json`; `reports/candidate_selection_audit.json`; `figures/final/`.', '', 'Exact commands and logs are in the registry. Source hashes and packages are in environment_report.json. Per-arm configs bind augmentation files and original frozen train/validation hashes. Canonical leakage TF-IDF char 3–5 representation is persisted; threshold 0.90 is unchanged. Curated count is the valid filtered count without duplicates. Upstream commit: 7984e2d6d95e98e914d2b0cd92574b1752614b10.']
    (REPORT/'FINAL_EXPERIMENT_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (REPORT/'PRIMARY_RESULTS.md').write_text(markdown_table(primary)+'\n',encoding='utf-8')
    return complete

def figures(frame,primary,ablation,categories):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    figdir=ROOT/'figures/final'
    figdir.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
    def save(fig,name):
        fig.tight_layout()
        for ext in ['png','pdf']:
            fig.savefig(figdir/f'{name}.{ext}',dpi=300,bbox_inches='tight')
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.2,3.1))
    labels=['Real Dolly + OASST\ncandidate prompts','Canonical leakage\nrepresentation (0.90)','Mean attack score\nbaseline seeds 13,42,123','Descending rank\nindependent semantic A+B','Nested Top-K / matched\nrandom benign augmentation','Controlled retraining\nfrozen PIDS evaluation']
    positions=[(1.2,1.85),(3.6,1.85),(6.,1.85),(6.,.65),(3.6,.65),(1.2,.65)]
    for (x,y),label in zip(positions,labels):
        ax.add_patch(FancyBboxPatch((x-1.02,y-.30),2.04,.60,boxstyle='round,pad=.04',facecolor='#e8f0ff',edgecolor='#2856a8'))
        ax.text(x,y,label,ha='center',va='center',fontsize=9)
    for (x,y),(u,v) in zip(positions,positions[1:]):
        if y==v:
            sign=1 if u>x else -1
            ax.annotate('',xy=(u-sign*1.08,v),xytext=(x+sign*1.08,y),arrowprops={'arrowstyle':'->','color':'#2856a8'})
        else:
            ax.annotate('',xy=(u,v+.36),xytext=(x,y-.36),arrowprops={'arrowstyle':'->','color':'#2856a8'})
    ax.set(xlim=(0,7.2),ylim=(0,2.5)); ax.axis('off')
    save(fig,'01_pipeline')
    colors=['#555555','#b67527','#35906c','#2856a8']
    names=['Baseline','Curated-HN','Random-Real-1000','ProvGuard-1000']
    for name,metric,label in [('02_primary_fpr','hard_benign_fpr','Usable hard-benign FPR (n=808)'),('03_external_fpr','external_hb_fpr','External hard-benign FPR (n=208)')]:
        fig,ax=plt.subplots(figsize=(8,4))
        x=np.arange(4)
        ax.bar(x,primary[metric+'_mean'],yerr=primary[metric+'_sd'].fillna(0),color=colors,capsize=4)
        ax.set(xticks=x,xticklabels=names,ylabel=label,ylim=(0,1))
        ax.tick_params(axis='x',rotation=15)
        save(fig,name)
    allrows=summary(frame,METHODS)
    fig,ax=plt.subplots(figsize=(7,4.5))
    for _,r in allrows.iterrows():
        ax.errorbar(r.hard_benign_fpr_mean,r.iid_recall_mean,xerr=r.hard_benign_fpr_sd,yerr=r.iid_recall_sd,fmt='o',capsize=3,label=r['method'])
    xmax=min(1,float((allrows.hard_benign_fpr_mean+allrows.hard_benign_fpr_sd).max())+.04)
    ymin=max(0,float((allrows.iid_recall_mean-allrows.iid_recall_sd).min())-.01)
    ax.set(xlabel='Usable hard-benign FPR (lower is better)',ylabel='IID attack recall (higher is better)',xlim=(0,xmax),ylim=(ymin,1.005))
    ax.legend(fontsize=8)
    save(fig,'04_security_usability')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for metric,label in [('hard_benign_fpr','HB FPR'),('external_hb_fpr','External HB FPR')]:
        axes[0].errorbar(ablation.K,ablation[metric+'_mean'],yerr=ablation[metric+'_sd'],fmt='o-',capsize=3,label=label)
    for metric,label in [('iid_recall','Attack recall'),('obfuscated_recall','Obfuscated recall'),('iid_f1','IID F1')]:
        axes[1].errorbar(ablation.K,ablation[metric+'_mean'],yerr=ablation[metric+'_sd'],fmt='o-',capsize=3,label=label)
    for ax in axes:
        ax.set(xlabel='Augmentation K',xticks=[250,500,1000],ylim=(0,1.02)); ax.legend(fontsize=8)
    axes[0].set_ylabel('False positive rate'); axes[1].set_ylabel('Detection metric')
    detection_low=min(float((ablation[m+'_mean']-ablation[m+'_sd']).min()) for m in ['iid_recall','obfuscated_recall','iid_f1'])
    axes[1].set_ylim(max(0,detection_low-.015),1.005)
    save(fig,'05_k_ablation')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for method,label,color in [('random_real_1000','Random-Real','#35906c'),('provguard_1000','ProvGuard','#2856a8')]:
        x=pd.read_csv(DATA/f'{method}.csv').mean_attack_probability.to_numpy()
        axes[0].hist(x,bins=np.linspace(0,1,41),density=True,histtype='step',linewidth=1.7,label=label,color=color)
        ordered=np.sort(x)
        axes[1].plot(ordered,np.arange(1,len(x)+1)/len(x),label=label,color=color)
    if (DATA/'curated_scored.csv').exists():
        x=pd.read_csv(DATA/'curated_scored.csv').mean_attack_probability.to_numpy()
        axes[0].hist(x,bins=np.linspace(0,1,41),density=True,histtype='step',linewidth=1.7,linestyle='--',label=f'Curated-HN (n={len(x)})',color='#b67527')
        ordered=np.sort(x)
        axes[1].plot(ordered,np.arange(1,len(x)+1)/len(x),linestyle='--',label=f'Curated-HN (n={len(x)})',color='#b67527')
    axes[0].set(xlabel='Mean baseline injection probability',ylabel='Density')
    axes[1].set(xlabel='Mean baseline injection probability',ylabel='Empirical cumulative proportion')
    for ax in axes:
        ax.legend(); ax.set_xlim(0,1)
    save(fig,'06_hardness_distribution')
    if len(categories):
        grouped=categories.groupby(['error_category','method']).fpr.agg(['mean','std'])
        cats=sorted(categories.error_category.unique())
        fig,ax=plt.subplots(figsize=(10,4.5))
        for i,(method,label) in enumerate([('baseline','Baseline'),('random_real','Random-Real'),('provguard_1000','ProvGuard')]):
            values=grouped.xs(method,level='method').reindex(cats)
            ax.bar(np.arange(len(cats))+(i-1)*.24,values['mean'],width=.24,yerr=values['std'],capsize=2,label=label)
        ax.set(xticks=np.arange(len(cats)),xticklabels=cats,ylabel='HB FPR by automated category',ylim=(0,1))
        ax.tick_params(axis='x',rotation=35)
        for label in ax.get_xticklabels():
            label.set_ha('right')
        ax.legend()
        save(fig,'07_error_categories')
    save_json(REPORT/'final_figure_provenance.json',{'results_master_sha256':sha(ROOT/'results_master.csv'),'primary_results_sha256':sha(REPORT/'PRIMARY_RESULTS.csv'),'ablation_results_sha256':sha(REPORT/'ABLATION_RESULTS.csv'),'candidate_selection_audit_sha256':sha(REPORT/'candidate_selection_audit.json'),'error_categories_sha256':sha(REPORT/'ERROR_CATEGORIES.csv'),'uncertainty':'sample SD across five seeds; not confidence intervals','figures':{p.name:sha(p) for p in sorted(figdir.glob('*'))}})

def export_results(final=False):
    REPORT.mkdir(exist_ok=True)
    frame=collect()
    primary=summary(frame,PRIMARY)
    ablation=summary(frame,['provguard_250','provguard_500','provguard_1000'])
    csv(primary,REPORT/'PRIMARY_RESULTS.csv')
    csv(ablation,REPORT/'ABLATION_RESULTS.csv')
    curated_hardness()
    if final:
        if not all((m,s) in set(zip(frame.method,frame.seed.astype(int))) for m,s in PLAN):
            raise ValueError('Final analysis requires all thirty completed required runs.')
        stats=statistics(frame)
        categories=error_analysis(frame)
        source_stratified(frame)
        comp=comparisons(frame)
        report(frame,primary,ablation,stats,comp,True)
        figures(frame,primary,ablation,categories)
    else:
        report(frame,primary,ablation,pd.DataFrame(columns=STAT_COLUMNS),pd.DataFrame(),False)
    return frame

if __name__=='__main__':
    export_results(final=True)
