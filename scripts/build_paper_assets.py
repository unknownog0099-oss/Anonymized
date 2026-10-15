"""Build final paper numbers, tables, and editable vector figures from frozen evidence.

Run from any directory: .venv/Scripts/python.exe scripts/build_paper_assets.py
No training, dataset selection, or experiment artifact is changed by this script.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import shutil
import textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / 'paper'
FIG = ROOT / 'figures' / 'paper'
TABLE = PAPER / 'tables'
COLORS = {'baseline':'#454B54', 'curated':'#C77D35', 'random_real':'#417DA8',
          'provguard_1000':'#218477', 'provguard_250':'#76A99A', 'provguard_500':'#427B65'}
LABELS = {'baseline':'Baseline', 'curated':'Curated-HN', 'random_real':'Random-Real',
          'provguard_1000':'ProvGuard', 'provguard_250':'ProvGuard-250', 'provguard_500':'ProvGuard-500'}
METHODS = list(LABELS)[:4]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9, 'axes.labelsize':9,
    'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'pdf.fonttype':42,
    'ps.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,
    'axes.linewidth':.7,'grid.linewidth':.5,'grid.color':'#DDE1E4','savefig.dpi':400})

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(name):
    return pd.read_csv(ROOT / 'reports' / name, keep_default_na=False)

def save(fig, name, inputs):
    for ext in ['pdf','png','svg']:
        fig.savefig(FIG/f'{name}.{ext}', bbox_inches='tight', pad_inches=.055,
                    metadata={'Creator':'ProvGuard evidence-based paper asset generator'} if ext=='pdf' else None)
    manifests[name] = {'source':'scripts/build_paper_assets.py', 'source_sha256':sha(Path(__file__)), 'input_sha256':{p:sha(ROOT/p) for p in inputs},
                        'files':{ext:sha(FIG/f'{name}.{ext}') for ext in ['pdf','png','svg']}}
    plt.close(fig)

def style(ax):
    ax.grid(axis='y',zorder=0)
    ax.set_axisbelow(True)

def box(ax,x,y,w,h,title,body,color='#218477',title_size=8,body_size=7.4):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.008,rounding_size=0.045',
                              lw=.85,ec=color,fc='white'))
    ax.text(x+w/2,y+h-.16,title,ha='center',va='center',weight='bold',fontsize=title_size,color=color)
    ax.text(x+w/2,y+h*.40,body,ha='center',va='center',fontsize=body_size,linespacing=1.3,color='#242A31')

def arrow(ax,a,b,color='#707780'):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=8,lw=.8,color=color))

def num(name,value,source,expression,fmt='.2f'):
    display = format(float(value),fmt) if fmt!='d' else str(int(value))
    macros.append('\\newcommand{\\'+name+'}{'+display+'}')
    ledger.append({'macro':name,'value':float(value),'display':display,'source':source,'expression':expression})

def token(name,display,value,source,expression):
    macros.append('\\newcommand{\\'+name+'}{'+display+'}')
    ledger.append({'macro':name,'value':value,'display':display,'source':source,'expression':expression})

def mean_sd(r,m):
    return f"{r[m+'_mean']:.4f}\\,\\pm\\,{r[m+'_sd']:.4f}"

def tabular(headers,rows,spec):
    return '\\begin{tabular}{'+spec+'}\n\\toprule\n'+' & '.join(headers)+' \\\\\n\\midrule\n'+''.join(' & '.join(row)+' \\\\\n' for row in rows)+'\\bottomrule\n\\end{tabular}\n'

def main():
    global manifests,macros,ledger
    manifests,macros,ledger = {}, ['% Generated from final canonical artifacts; do not hand-edit.'], []
    FIG.mkdir(parents=True,exist_ok=True); TABLE.mkdir(parents=True,exist_ok=True)
    primary=read('PRIMARY_RESULTS.csv').set_index('method')
    ablation=read('ABLATION_RESULTS.csv').set_index('method')
    statistics=read('STATISTICAL_RESULTS.csv')
    stats=statistics.loc[statistics.seed.astype(str)=='AGGREGATE']
    hard=read('HARDNESS_ANALYSIS.csv').set_index('method')
    comparison=read('COMPARISON_RESULTS.csv').set_index('comparison')
    protocol=json.loads((ROOT/'configs/protocol.json').read_text())
    leak=json.loads((ROOT/'reports/canonical_leakage_audit.json').read_text())
    review=json.loads((ROOT/'reports/semantic_adjudication_audit.json').read_text())
    select=json.loads((ROOT/'reports/candidate_selection_audit.json').read_text())
    verdict=json.loads((ROOT/'reports/hypothesis_verdicts.json').read_text())
    registry_path=ROOT/'reports/experiment_registry.json'
    if not registry_path.exists():registry_path=ROOT/'artifacts/experiment_registry.json'
    registry=json.loads(registry_path.read_text())['experiments']
    assert len(registry)==30 and all(r['status']=='COMPLETED' for r in registry)
    assert all(primary.seed_n==5) and all(ablation.seed_n==5)
    assert all(primary.status=='COMPLETE') and all(ablation.status=='COMPLETE')
    assert protocol['primary_k']==1000
    metrics={'IID':'iid_f1','Rec':'iid_recall','HB':'hard_benign_fpr','Ext':'external_hb_fpr',
             'Obf':'obfuscated_recall','Dom':'domain_ood_f1','Struct':'structural_ood_fpr'}
    for method,prefix in [('baseline','Base'),('curated','Cur'),('random_real','Rand'),('provguard_1000','PG')]:
        for short,m in metrics.items():
            num(prefix+short,primary.loc[method,m+'_mean'],'reports/PRIMARY_RESULTS.csv',method+'/'+m+'_mean','.4f')
            num(prefix+short+'Pct',100*primary.loc[method,m+'_mean'],'reports/PRIMARY_RESULTS.csv','100 * '+method+'/'+m+'_mean')
    for method,prefix in [('curated','Cur'),('provguard_1000','PG')]:
        num(prefix+'CuratedFPR',primary.loc[method,'curated_hb_fpr_mean'],'reports/PRIMARY_RESULTS.csv',method+'/curated_hb_fpr_mean','.4f')
    for comparator,prefix in [('baseline','Base'),('random_real','Rand'),('curated','Cur')]:
        c=comparison.loc[comparator+' vs provguard_1000']
        for short,m in [('HB','hard_benign'),('Ext','external_hb')]:
            num(prefix+short+'DropPP',100*c[m+'_fpr_absolute_reduction'],'reports/COMPARISON_RESULTS.csv',comparator+'/'+m+' absolute delta * 100')
            num(prefix+short+'DropRel',100*c[m+'_fpr_relative_reduction'],'reports/COMPARISON_RESULTS.csv',comparator+'/'+m+' relative delta * 100','.1f')
        for short,m in [('IID','iid_f1'),('Rec','iid_recall'),('Dom','domain_ood_f1'),('Struct','structural_ood_fpr')]:
            num(prefix+short+'ChangePP',100*c[m+'_provguard_minus_comparator'],'reports/COMPARISON_RESULTS.csv',comparator+'/'+m+' change * 100','+.2f')
    for _,r in stats.iterrows():
        prefix={'baseline':'Base','random_real':'Rand','curated':'Cur'}[r.comparison.split(' vs ')[0]]
        suffix='ExtCI' if r.subset=='external_hard_benign' else 'HBCI'
        display=f"[{r.bootstrap_ci_lower:.4f},\\,{r.bootstrap_ci_upper:.4f}]"
        token(prefix+suffix,display,[r.bootstrap_ci_lower,r.bootstrap_ci_upper],'reports/STATISTICAL_RESULTS.csv',r.comparison+'/'+r.subset+'/AGGREGATE 95% CI')
    for short,m in [('IID','iid_f1'),('Rec','iid_recall'),('Obf','obfuscated_recall')]:
        r=verdict['attack_preservation'][m]
        display=f"[{r['seed_bootstrap_ci_lower']:.4f},\\,{r['seed_bootstrap_ci_upper']:.4f}]"
        token('Preserve'+short+'CI',display,[r['seed_bootstrap_ci_lower'],r['seed_bootstrap_ci_upper']],
              'reports/hypothesis_verdicts.json','attack_preservation/'+m+' paired-seed CI')
    counts={'NTrain':leak['benchmark_integrity']['train']['usable_text_n'],'NVal':leak['benchmark_integrity']['val']['usable_text_n'],
        'NIID':leak['benchmark_integrity']['test']['usable_text_n'],'NHB':leak['benchmark_integrity']['hard_benign']['usable_text_n'],
        'NReleasedHB':leak['benchmark_integrity']['hard_benign']['rows'],'NWithheld':leak['benchmark_integrity']['hard_benign']['missing_text_n'],
        'NObf':leak['benchmark_integrity']['obfuscated']['usable_text_n'],'NDomain':leak['benchmark_integrity']['domain_ood']['usable_text_n'],
        'NStruct':leak['benchmark_integrity']['structural_ood']['usable_text_n'],'NCandidatesBefore':leak['real']['before_n'],
        'NCandidates':leak['real']['final_n'],'NExactRemoved':leak['real']['exact_removed'],'NNearRemoved':leak['real']['near_removed'],
        'NCurAug':leak['curated']['final_n'],'NReviewed':review['paired_review_n'],'NApproved':review['final_counts']['BENIGN'],
        'NRejected':review['final_counts']['INJECTION'],'NAmbiguous':review['final_counts']['AMBIGUOUS'],
        'NDollySelected':select['selected_source_counts']['dolly'],'NOasstSelected':select['selected_source_counts']['oasst1'],
        'NOverlap':select['random_overlap_n'],'NLastRank':select['rank_complete_through']}
    for name,value in counts.items():
        source='reports/semantic_adjudication_audit.json' if name in ['NReviewed','NApproved','NRejected','NAmbiguous'] else 'reports/candidate_selection_audit.json' if name in ['NDollySelected','NOasstSelected','NOverlap','NLastRank'] else 'reports/canonical_leakage_audit.json'
        num(name,value,source,name,'d')
    master=pd.read_csv(ROOT/'results_master.csv')
    import sys
    sys.path.insert(0,str(ROOT/'src'))
    from provguard.core import read_texts,BENCH
    hb=read_texts(BENCH/'eval_subsets/hard_benign_test.csv')
    external=hb.loc[hb.source.astype(str).str.startswith(('dolly','oasst1'))]
    assert len(hb)==808 and len(external)==208
    num('NExternal',len(external),'upstream/data/pids_bench_v3/eval_subsets/hard_benign_test.csv','usable source dolly/oasst1','d')
    num('NCuratedHB',len(hb)-len(external),'upstream/data/pids_bench_v3/eval_subsets/hard_benign_test.csv','usable non-external rows','d')
    for name,value,key in [('PrimaryK',protocol['primary_k'],'primary_k'),('NSeeds',len(protocol['seeds']),'len(seeds)'),
        ('NSelectors',len(protocol['selector_seeds']),'len(selector_seeds)'),('MinChars',protocol['candidate_min_chars'],'candidate_min_chars'),
        ('NEpochs',protocol['epochs'],'epochs'),('NBatch',protocol['batch_size'],'batch_size'),('MaxLength',protocol['max_length'],'max_length')]:
        num(name,value,'configs/protocol.json',key,'d')
    num('NRuns',len(registry),registry_path.relative_to(ROOT).as_posix(),'count verified COMPLETED rows','d')
    num('Threshold',protocol['threshold'],'configs/protocol.json','threshold','.1f')
    num('LeakThreshold',protocol['near_duplicate_threshold'],'configs/protocol.json','near_duplicate_threshold','.2f')
    num('WeightDecay',protocol['weight_decay'],'configs/protocol.json','weight_decay','.2f')
    num('MaxGradNorm',protocol['max_gradient_norm'],'configs/protocol.json','max_gradient_norm','.1f')
    num('BootstrapN',stats.bootstrap_repetitions.iloc[0],'reports/STATISTICAL_RESULTS.csv','bootstrap_repetitions','d')
    for method,prefix in [('provguard_1000','HardPG'),('random_real_1000','HardRand'),('curated','HardCur')]:
        for short,key in [('Mean','mean'),('Median','median'),('QThree','q3')]:
            num(prefix+short,hard.loc[method,key],'reports/HARDNESS_ANALYSIS.csv',method+'/'+key,'.6f')
        num(prefix+'FlagN',hard.loc[method,'n_above_0.5'],'reports/HARDNESS_ANALYSIS.csv',method+'/n_above_0.5','d')
    for method,prefix in [('provguard_250','KSmall'),('provguard_500','KMedium')]:
        for short,m in [('HB','hard_benign_fpr'),('Ext','external_hb_fpr')]:
            num(prefix+short+'Pct',100*ablation.loc[method,m+'_mean'],'reports/ABLATION_RESULTS.csv',method+'/'+m+'_mean * 100')
    secondary=pd.read_csv(ROOT/'reports/SECONDARY_RESULTS.csv')
    for method,prefix in [('provguard_dolly_250','SourceDolly'),('provguard_oasst1_250','SourceOasst')]:
        g=secondary.loc[secondary.method==method]
        for short,metric in [('Dolly','dolly_fpr'),('Oasst','oasst1_fpr')]:
            num(prefix+short+'FPR',g[metric].mean(),'reports/SECONDARY_RESULTS.csv',method+'/'+metric+' mean','.4f')
    num('SourceSeedN',secondary.loc[secondary.method=='provguard_dolly_250','seed'].nunique(),'reports/SECONDARY_RESULTS.csv','source arm seed_n','d')
    for name,metric in [('NDollyExternal','dolly_n'),('NOasstExternal','oasst1_n')]:
        assert secondary[metric].nunique()==1
        num(name,secondary[metric].iloc[0],'reports/SECONDARY_RESULTS.csv',metric,'d')
    (PAPER/'results_macros.tex').write_text('\n'.join(macros)+'\n',encoding='utf-8')
    (PAPER/'numerical_claims.json').write_text(json.dumps(ledger,indent=2),encoding='utf-8')
    shutil.copyfile(ROOT/'results_master.csv',PAPER/'results_snapshot.csv')

    # Exact technical tables; bold is determined by the actual direction of each metric.
    ms=list(metrics.values()); low={'hard_benign_fpr','external_hb_fpr','structural_ood_fpr'}
    headers=['Method','Aug. $N$','\\shortstack{IID F1\\\\$\\uparrow$}','\\shortstack{Attack recall\\\\$\\uparrow$}',
        '\\shortstack{HB FPR\\\\$\\downarrow$}','\\shortstack{External FPR\\\\$\\downarrow$}',
        '\\shortstack{Obf. recall\\\\$\\uparrow$}','\\shortstack{Domain F1\\\\$\\uparrow$}','\\shortstack{Struct. FPR\\\\$\\downarrow$}']
    rows=[]
    for method in METHODS:
        r=primary.loc[method]; cells=[]
        for m in ms:
            val=mean_sd(r,m)
            best=primary[m+'_mean'].min() if m in low else primary[m+'_mean'].max()
            cells.append(('\\textbf{'+f'{r[m+"_mean"]:.4f}'+'}'+f'$\\,\\pm\\,{r[m+"_sd"]:.4f}$') if r[m+'_mean']==best else '$'+val+'$')
        rows.append([LABELS[method],str(int(r.augmentation_size))]+cells)
    (TABLE/'primary.tex').write_text(tabular(headers,rows,'@{}lr'+'c'*7+'@{}'),encoding='utf-8')
    rows=[]
    for comparator in ['baseline','random_real','curated']:
        for subset in ['usable_hard_benign','external_hard_benign']:
            r=stats.loc[(stats.comparison==comparator+' vs provguard_1000')&(stats.subset==subset)].iloc[0]
            interpretation='PG lower' if r.bootstrap_ci_lower>0 else 'Curated lower' if r.bootstrap_ci_upper<0 else 'Inconclusive'
            rows.append([LABELS[comparator], 'Overall' if subset.startswith('usable') else 'External',
                         f'${r.delta_FPR:+.4f}$',f'$[{r.bootstrap_ci_lower:.4f},\,{r.bootstrap_ci_upper:.4f}]$',interpretation])
    (TABLE/'statistics.tex').write_text(tabular(['Comparator','Subset','$\\Delta$FPR','95\\% CI','Interpretation'],rows,'@{}llrrl@{}'),encoding='utf-8')
    rows=[]
    for method in ['provguard_250','provguard_500','provguard_1000']:
        r=ablation.loc[method]
        rows.append([str(int(r.K))]+['$'+mean_sd(r,m)+'$' for m in ['iid_f1','iid_recall','hard_benign_fpr','external_hb_fpr','obfuscated_recall']])
    (TABLE/'ablation.tex').write_text(tabular(['$K$','IID F1','Attack recall','HB FPR','External FPR','Obf. recall'],rows,'@{}rccccc@{}'),encoding='utf-8')
    rows=[]
    for method in ['provguard_1000','random_real_1000','curated']:
        r=hard.loc[method]; rows.append([{'provguard_1000':'ProvGuard','random_real_1000':'Random-Real','curated':'Curated-HN'}[method],str(int(r.n))]+[f'{r[k]:.6f}' for k in ['mean','median','q3']]+[str(int(r['n_above_0.5']))])
    (TABLE/'hardness.tex').write_text(tabular(['Method','$N$','Mean','Median','$Q_3$','$h>0.5$'],rows,'@{}lrrrrr@{}'),encoding='utf-8')
    rows=[]
    for _,r in master.iterrows():
        rows.append([LABELS[r.method],str(int(r.seed)),str(int(r.augmentation_size))]+[f'{r[m]:.4f}' for m in ms])
    (TABLE/'per_seed.tex').write_text(tabular(['Method','Seed','Aug. $N$','IID F1','Attack recall','HB FPR','External FPR','Obf. recall','Domain F1','Struct. FPR'],rows,'@{}lrr'+'r'*7+'@{}'),encoding='utf-8')
    sec=read('SECONDARY_RESULTS.csv'); rows=[]
    for method,label in [('provguard_dolly_250','Dolly-only augmentation'),('provguard_oasst1_250','OASST-only augmentation')]:
        g=sec.loc[sec.method==method]
        rows.append([label,str(len(g))]+[f'${g[m].mean():.4f}\\,\\pm\\,{g[m].std(ddof=1):.4f}$' for m in ['iid_f1','iid_recall','dolly_fpr','oasst1_fpr']])
    (TABLE/'source_transfer.tex').write_text(tabular(['Augmentation','Seeds','IID F1','Attack recall','Dolly FPR','OASST FPR'],rows,'@{}lrcccc@{}'),encoding='utf-8')
    # Error evidence is derived by row identity, not hand-selected favorable counts.
    transitions=read('ERROR_TRANSITIONS.csv')
    fp=transitions.loc[transitions.subset=='hard_benign'].groupby(['seed','transition_type']).size().unstack(fill_value=0)
    for name,key in [('FixedFP','FP_FIXED'),('NewFP','NEW_FP')]:
        num(name,fp.get(key,pd.Series(0,index=fp.index)).mean(),'reports/ERROR_TRANSITIONS.csv',key+' mean count per seed',' .1f'.strip())
    # num() added error macros after the initial write; persist the complete ledger.
    (PAPER/'results_macros.tex').write_text('\n'.join(macros)+'\n',encoding='utf-8')
    (PAPER/'numerical_claims.json').write_text(json.dumps(ledger,indent=2),encoding='utf-8')
    score=pd.read_csv(ROOT/'data/provguard/provguard_1000.csv',keep_default_na=False)
    queue=score.loc[score.mean_attack_probability>=.5].copy()
    human_cols=['human_decision','human_reason','human_reviewer']
    queue_path=ROOT/'reports/MANUAL_AUDIT_QUEUE.csv'
    if queue_path.exists():
        old=pd.read_csv(queue_path,keep_default_na=False)
        if set(human_cols).issubset(old.columns) and not old.candidate_id.duplicated().any():
            queue=queue.merge(old[['candidate_id']+human_cols],on='candidate_id',how='left',validate='one_to_one')
    for c in human_cols:
        queue[c]=queue[c].fillna('') if c in queue else ''
    queue.to_csv(queue_path,index=False)
    assert len(queue)==172

    # 1. Vector methodology diagram, readable at the IEEE full-text width.
    fig,ax=plt.subplots(figsize=(7.15,2.75)); ax.set_xlim(0,7.15);ax.set_ylim(0,2.75);ax.axis('off')
    ax.text(.04,2.63,'PROVGUARD  |  ONLY TRAINING AUGMENTATION CHANGES',fontsize=9,weight='bold',color='#242A31')
    xs=np.arange(5)*1.44+.03;w=1.31;ht=.90;yt=1.54;yb=.27
    nodes=[('Sources','Dolly instructions\nOASST1 English roots'),('Clean','Nonempty; >=20 chars\nNo corruption\nInternal exact dedup'),
           ('Lexical leakage','All frozen references\nExact + char 3-5 TF-IDF\nCosine >=0.90 removed'),
           ('Selector ensemble','Baseline DistilBERT\nSeeds 13 / 42 / 123'),('Score and rank','Mean injection score h(x)\nRank descending h(x)\nCandidate-ID tie break')]
    lower=[('Semantic validity','Two separate passes\nBoth BENIGN: keep\nOtherwise: exclude'),('Nested Top-K','250 / 500 / 1000\nK=1000 primary'),
           ('Augment training','Original PIDS train\n+ selected benign data\nAugmentation label y=0'),('Same detector','DistilBERT; fixed setup\nOriginal-val checkpoint\nThreshold 0.5'),
           ('Frozen evaluation','IID; HB; external HB\nObfuscated attacks\nDomain / structural OOD')]
    for i,(title,body) in enumerate(nodes):box(ax,xs[i],yt,w,ht,title,body,color='#454B54' if i<3 else '#218477')
    for i,(title,body) in enumerate(lower):box(ax,xs[i],yb,w,ht,title,body)
    for y in [yt,yb]:
        for i in range(4):arrow(ax,(xs[i]+w+.01,y+ht/2),(xs[i+1]-.012,y+ht/2))
    # Routed return connector between rows keeps both rows left-to-right.
    ax.plot([xs[-1]+w/2,xs[-1]+w/2,xs[0]+w/2],[yt,1.35,1.35],color='#707780',lw=.8)
    arrow(ax,(xs[0]+w/2,1.35),(xs[0]+w/2,yb+ht+.02))
    ax.text(.04,.08,f"{leak['real']['final_n']:,} leakage-filtered real candidates; matched random control uses the same source/length quotas.",fontsize=8,color='#454B54')
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    save(fig,'methodology',['configs/protocol.json','reports/canonical_leakage_audit.json','reports/semantic_adjudication_audit.json','reports/candidate_selection_audit.json'])

    # 2. Main grouped result plot.
    fig,ax=plt.subplots(figsize=(7.15,3)); xx=np.arange(2);bw=.17
    for i,method in enumerate(METHODS):
        r=primary.loc[method]; vals=[100*r.hard_benign_fpr_mean,100*r.external_hb_fpr_mean]
        sds=[100*r.hard_benign_fpr_sd,100*r.external_hb_fpr_sd]
        ax.bar(xx+(i-1.5)*bw,vals,bw,yerr=sds,color=COLORS[method],label=LABELS[method],
               edgecolor='white',linewidth=.5,capsize=2.5,zorder=3,hatch=['','//','..','xx'][i],error_kw={'lw':.8})
    ax.set_xticks(xx,['Overall hard-benign\n600 curated + 208 external','External hard-benign\n208 public-text prompts'])
    ax.set_ylabel('False-positive rate (%)');ax.set_ylim(0,58);style(ax)
    ax.legend(ncol=4,loc='upper center',bbox_to_anchor=(.5,1.15),frameon=False)
    fig.subplots_adjust(left=.085,right=.99,bottom=.22,top=.88)
    save(fig,'primary_results',['reports/PRIMARY_RESULTS.csv'])

    # 3. Dedicated target-domain result.
    fig,ax=plt.subplots(figsize=(3.5,2.50)); xx=np.arange(4)
    values=100*primary.loc[METHODS,'external_hb_fpr_mean'].to_numpy();sds=100*primary.loc[METHODS,'external_hb_fpr_sd'].to_numpy()
    bars=ax.bar(xx,values,.62,yerr=sds,color=[COLORS[m] for m in METHODS],capsize=3,zorder=3,error_kw={'lw':.8})
    for i,(bar,v,s) in enumerate(zip(bars,values,sds)):
        bar.set_hatch(['','//','..','xx'][i]);bar.set_edgecolor('white')
        ax.text(i,v+s+.6,f'{v:.2f}',ha='center',fontsize=8,weight='bold' if i==3 else 'normal')
    ax.set_xticks(xx,['Baseline','Curated-HN','Random-\nReal','ProvGuard'])
    ax.set_ylabel('External HB\nfalse-positive rate (%)');ax.set_ylim(0,29);style(ax)
    ax.text(.97,.96,f"{100*comparison.loc['baseline vs provguard_1000','external_hb_fpr_relative_reduction']:.1f}% reduction vs baseline",
            ha='right',va='top',transform=ax.transAxes,fontsize=8,color=COLORS['provguard_1000'])
    fig.subplots_adjust(left=.17,right=.99,bottom=.23,top=.98)
    save(fig,'external_hb',['reports/PRIMARY_RESULTS.csv','reports/COMPARISON_RESULTS.csv'])

    # 4. Security-usability diagnostic at one fixed threshold, not a ROC curve.
    fig,ax=plt.subplots(figsize=(7.15,3.25))
    points=pd.concat([primary,ablation.loc[['provguard_250','provguard_500']]])
    markers={'baseline':'s','curated':'^','random_real':'o','provguard_250':'v','provguard_500':'P','provguard_1000':'D'}
    for method in ['baseline','curated','random_real','provguard_250','provguard_500','provguard_1000']:
        r=points.loc[method]
        ax.errorbar(100*r.hard_benign_fpr_mean,100*r.iid_recall_mean,xerr=100*r.hard_benign_fpr_sd,yerr=100*r.iid_recall_sd,
                    fmt=markers[method],ms=7 if method=='provguard_1000' else 6,color=COLORS[method],capsize=3,lw=.9,
                    label='ProvGuard-1000 (primary)' if method=='provguard_1000' else LABELS[method],alpha=.95)
    ax.set_xlabel('Overall hard-benign false-positive rate (%)  [lower is better]')
    ax.set_ylabel('IID attack recall (%)\n[higher is better]');ax.set_xlim(0,57);ax.set_ylim(97.5,99.4);style(ax)
    ax.text(.02,.96,'Preferred direction: upper left',transform=ax.transAxes,fontsize=8,color='#454B54',va='top')
    ax.legend(ncol=3,frameon=False,loc='upper center',bbox_to_anchor=(.5,-.20))
    fig.subplots_adjust(left=.10,right=.99,bottom=.28,top=.98)
    save(fig,'security_usability',['reports/PRIMARY_RESULTS.csv','reports/ABLATION_RESULTS.csv'])

    # 5. Non-monotonic K ablation, common percentage axis.
    fig,ax=plt.subplots(figsize=(3.5,2.50)); ks=[250,500,1000]; r=ablation.loc[[f'provguard_{k}' for k in ks]]
    for metric,color,marker,linestyle,label in [('hard_benign_fpr','#218477','o','-','Overall HB'),('external_hb_fpr','#35585A','s','--','External HB')]:
        ax.errorbar(ks,100*r[metric+'_mean'],yerr=100*r[metric+'_sd'],color=color,marker=marker,ls=linestyle,capsize=3,lw=1.15,label=label,ms=4)
    ax.set_xticks(ks,['250','500','1000\n(primary)']);ax.set_xlabel('Augmentation size K');ax.set_ylabel('False-positive rate (%)');ax.set_ylim(0,29);style(ax)
    ax.legend(frameon=False,loc='upper left',ncol=2,fontsize=8)
    fig.subplots_adjust(left=.17,right=.99,bottom=.27,top=.98)
    save(fig,'ablation',['reports/ABLATION_RESULTS.csv'])

    # 6. ECDF preserves the highly skewed hardness distribution.
    fig,ax=plt.subplots(figsize=(3.5,2.50))
    score_inputs=[]
    for file,method,label,ls in [('data/provguard/provguard_1000.csv','provguard_1000','ProvGuard','-'),
        ('data/provguard/random_real_1000.csv','random_real','Random-Real','--'),
        ('data/provguard/curated_scored.csv','curated','Curated-HN',':')]:
        d=pd.read_csv(ROOT/file);v=np.sort(d.mean_attack_probability.to_numpy())
        ax.step(np.r_[v[0],v],np.r_[0,np.arange(1,len(v)+1)/len(v)],where='post',color=COLORS[method],ls=ls,lw=1.25,label=label)
        score_inputs.append(file)
    ax.axvline(.5,color='#969DA5',ls='--',lw=.8,zorder=0);ax.text(.49,.06,'0.5',ha='right',fontsize=8,color='#6D747B')
    ax.set_xscale('log');ax.set_xlim(.0002,1.05);ax.set_ylim(0,1.02)
    ax.set_xlabel('Selector injection probability (log scale)');ax.set_ylabel('Cumulative proportion');style(ax)
    ax.legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,1.20),ncol=3,fontsize=8,handlelength=1.3,columnspacing=.7)
    fig.subplots_adjust(left=.17,right=.99,bottom=.22,top=.84)
    save(fig,'hardness_distribution',score_inputs+['reports/HARDNESS_ANALYSIS.csv'])

    # 7. Standalone graphical abstract with a real, labeled observed example.
    example=transitions.loc[(transitions.subset=='hard_benign')&(transitions.row_id=='hard_benign_test:1059')]
    assert len(example)==5 and example.transition_type.eq('FP_FIXED').all()
    fig=plt.figure(figsize=(12.2,5.5));ax=fig.add_axes([0,0,1,1]);ax.set_xlim(0,12.2);ax.set_ylim(0,5.5);ax.axis('off')
    ax.text(.35,5.10,'PROVGUARD',fontsize=21,weight='bold',color='#218477')
    ax.text(3.2,5.12,'Confusion-selected benign data for prompt-injection detectors',fontsize=14,color='#242A31')
    for x in [3.88,7.89]:ax.plot([x,x],[.70,4.66],lw=.8,color='#DDE1E4')
    ax.text(.35,4.45,'01  THE FAILURE',fontsize=11,weight='bold',color=COLORS['baseline'])
    ax.text(.35,4.02,'High IID accuracy can hide\nbenign over-defense',fontsize=16,linespacing=1.2,color='#242A31')
    ax.text(.35,3.22,f"{100*primary.loc['baseline','iid_f1_mean']:.2f}%",fontsize=27,weight='bold',color=COLORS['baseline'])
    ax.text(.35,2.91,'IID F1',fontsize=11,color='#454B54')
    ax.text(2.05,3.22,f"{100*primary.loc['baseline','external_hb_fpr_mean']:.2f}%",fontsize=27,weight='bold',color=COLORS['baseline'])
    ax.text(2.05,2.91,'External benign FPR',fontsize=11,color='#454B54')
    ax.text(.35,2.42,'Observed benign stress example',fontsize=10,weight='bold',color='#454B54')
    ax.text(.35,2.19,textwrap.fill(example.text.iloc[0],40),fontsize=10,va='top',linespacing=1.4,color='#242A31')
    ax.text(.35,1.20,'Flagged by baseline: 5/5 seeds\nFlagged by ProvGuard: 0/5 seeds',fontsize=11,linespacing=1.4,color='#218477')
    ax.text(.35,.80,'Released curated example; not an external-set exemplar.',fontsize=8.8,color='#6D747B')
    ax.text(4.13,4.45,'02  TARGETED TRAINING DATA',fontsize=11,weight='bold',color='#218477')
    stages=[('Real candidate prompts','Dolly + OASST1; lexical leakage controls'),
            ('Baseline ensemble scoring','Mean injection probability; seeds 13 / 42 / 123'),
            ('Rank + semantic validity','Two separate automated passes; both BENIGN'),
            ('Top-K + retrain same detector','K=1000 primary; no architecture change')]
    for i,(title,body) in enumerate(stages):
        y=3.48-i*.80;box(ax,4.15,y,3.46,.65,title,body,title_size=11,body_size=9.2)
        if i<3:arrow(ax,(5.88,y-.03),(5.88,y-.15),color='#218477')
    ax.text(4.15,.79,'Fixed threshold 0.5; five training seeds per arm.',fontsize=9.3,color='#454B54')
    ax.text(8.16,4.45,'03  EXTERNAL-SUBSET RESULT',fontsize=11,weight='bold',color='#218477')
    chart=fig.add_axes([.752,.415,.227,.34]);yy=np.arange(4)
    chart.barh(yy,values,color=[COLORS[m] for m in METHODS],xerr=sds,height=.56,capsize=3,error_kw={'lw':.8})
    chart.invert_yaxis();chart.set_yticks(yy,[LABELS[m] for m in METHODS],fontsize=10);chart.set_xlim(0,29)
    chart.set_xlabel('External HB FPR (%)',fontsize=10);chart.tick_params(axis='x',labelsize=9)
    for i,(v,s) in enumerate(zip(values,sds)):chart.text(v+s+.55,i,f'{v:.2f}%',va='center',fontsize=10,weight='bold' if i==3 else 'normal')
    chart.spines[['left','top','right']].set_visible(False);chart.grid(axis='x',lw=.5);chart.set_axisbelow(True)
    ax.text(8.16,1.55,f"Attack recall: {100*primary.loc['baseline','iid_recall_mean']:.2f}% -> {100*primary.loc['provguard_1000','iid_recall_mean']:.2f}%",fontsize=12,weight='bold',color='#242A31')
    ax.text(8.16,1.15,'Random-Real external gain supported;\nmixed-set advantage remains inconclusive.',fontsize=10,color='#454B54',linespacing=1.4)
    ax.text(.35,.32,'Targeted real hard negatives reduce external over-defense while retaining attack detection.',fontsize=14,weight='bold',color='#218477')
    ax.text(8.16,.79,'N=208 external prompts; bars show mean +/- sample SD.',fontsize=8.8,color='#6D747B')
    save(fig,'graphical_abstract',['reports/PRIMARY_RESULTS.csv','reports/STATISTICAL_RESULTS.csv','reports/ERROR_TRANSITIONS.csv'])
    manifests['graphical_abstract']['example']={'row_id':'hard_benign_test:1059','source':'released curated','baseline_flagged_seeds':5,'provguard_flagged_seeds':0}
    (FIG/'asset_manifest.json').write_text(json.dumps(manifests,indent=2),encoding='utf-8')
    print('Built 7 vector PDF/SVG figures, 400 dpi PNGs, 6 table sources, numeric claim ledger, and 172-row unreviewed human-audit queue.')

if __name__=='__main__':
    main()
