"""Read-only verification of experiment evidence, paper numbers, and LaTeX output.

Only the paper audit files are written. --deep independently checks all completed
models/predictions and the persisted leakage representation; it never refits it.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import sys
import numpy as np
import pandas as pd
import pymupdf as fitz

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
PAPER=ROOT/'paper'
METRICS=['iid_f1','iid_recall','hard_benign_fpr','external_hb_fpr','obfuscated_recall','domain_ood_f1','structural_ood_fpr']
METHODS=['baseline','curated','random_real','provguard_1000','provguard_250','provguard_500']
SEEDS=[13,42,123,2024,7777]

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):return json.loads((ROOT/path).read_text(encoding='utf-8'))
def close(a,b):assert np.allclose(a,b,rtol=0,atol=2e-14),(a,b)
def numbers(path):return [float(x) for x in re.findall(r'(?<![\w.])[+-]?\d+\.\d{4,6}(?!\d)',Path(path).read_text())]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--deep',action='store_true');args=parser.parse_args()
    master=pd.read_csv(ROOT/'results_master.csv');p=pd.read_csv(ROOT/'reports/PRIMARY_RESULTS.csv').set_index('method')
    a=pd.read_csv(ROOT/'reports/ABLATION_RESULTS.csv').set_index('method');s=pd.read_csv(ROOT/'reports/STATISTICAL_RESULTS.csv',keep_default_na=False)
    aggregate=s.loc[s.seed=='AGGREGATE']; protocol=load('configs/protocol.json')
    registry=load('reports/experiment_registry.json' if (ROOT/'reports/experiment_registry.json').exists() else 'artifacts/experiment_registry.json')['experiments'];select=load('reports/candidate_selection_audit.json')
    assert len(registry)==30 and all(r['status']=='COMPLETED' for r in registry)
    assert set(zip(master.method,master.seed))=={(m,k) for m in METHODS for k in SEEDS}
    assert len(master)==30 and master.threshold.eq(.5).all()
    summary_cells=0
    for method in METHODS:
        table=p if method in p.index else a;g=master.loc[master.method==method]
        assert set(g.seed)==set(SEEDS)
        for metric in METRICS+['curated_hb_fpr']:
            close(g[metric].mean(),table.loc[method,metric+'_mean']);close(g[metric].std(ddof=1),table.loc[method,metric+'_sd'])
            summary_cells+=2
    # Ordered literal values in every table are checked independently of the writer.
    wanted=[]
    for m in METHODS[:4]:
        for metric in METRICS:wanted.extend([p.loc[m,metric+'_mean'],p.loc[m,metric+'_sd']])
    close(numbers(PAPER/'tables/primary.tex'),np.round(wanted,4))
    wanted=[]
    for m in ['provguard_250','provguard_500','provguard_1000']:
        for metric in ['iid_f1','iid_recall','hard_benign_fpr','external_hb_fpr','obfuscated_recall']:
            wanted.extend([a.loc[m,metric+'_mean'],a.loc[m,metric+'_sd']])
    close(numbers(PAPER/'tables/ablation.tex'),np.round(wanted,4))
    wanted=[]
    for comparator in ['baseline','random_real','curated']:
        for subset,metric in [('usable_hard_benign','hard_benign_fpr'),('external_hard_benign','external_hb_fpr')]:
            row=aggregate.loc[(aggregate.comparison==comparator+' vs provguard_1000')&(aggregate.subset==subset)].iloc[0]
            close(row.delta_FPR,p.loc[comparator,metric+'_mean']-p.loc['provguard_1000',metric+'_mean'])
            assert row.training_seed_n==5 and row.bootstrap_repetitions==10000
            wanted.extend([row.delta_FPR,row.bootstrap_ci_lower,row.bootstrap_ci_upper])
    close(numbers(PAPER/'tables/statistics.tex'),np.round(wanted,4))
    hardness=pd.read_csv(ROOT/'reports/HARDNESS_ANALYSIS.csv').set_index('method');wanted=[]
    for m in ['provguard_1000','random_real_1000','curated']:
        wanted.extend(hardness.loc[m,['mean','median','q3']].to_numpy(dtype=float))
    close(numbers(PAPER/'tables/hardness.tex'),np.round(wanted,6))
    close(numbers(PAPER/'tables/per_seed.tex'),np.round(master[METRICS].to_numpy().ravel(),4))
    secondary=pd.read_csv(ROOT/'reports/SECONDARY_RESULTS.csv');wanted=[]
    for m in ['provguard_dolly_250','provguard_oasst1_250']:
        g=secondary.loc[secondary.method==m]
        for metric in ['iid_f1','iid_recall','dolly_fpr','oasst1_fpr']:wanted.extend([g[metric].mean(),g[metric].std(ddof=1)])
    close(numbers(PAPER/'tables/source_transfer.tex'),np.round(wanted,4))

    # Prose numbers are bound to a generated claim ledger, not copied from a prompt.
    ledger=load('paper/numerical_claims.json')
    manuscript=(ROOT/'ProvGuard.tex').read_text(encoding='utf-8');used=[]
    inline='% BEGIN INLINE NUMERICAL MACROS' in manuscript
    macro_text=(PAPER/'results_macros.tex').read_text()
    if inline:
        assert manuscript.split('% BEGIN INLINE NUMERICAL MACROS\n',1)[1].split('% END INLINE NUMERICAL MACROS',1)[0]==macro_text
        assert not re.search(r'\\(?:input|include|includegraphics|bibliography|bibliographystyle)\s*[\[{]',manuscript)
        for name in ['primary','statistics','ablation']:
            key='paper/tables/'+name
            body=manuscript.split('% BEGIN INLINE TABLE '+key+'\n',1)[1].split('% END INLINE TABLE '+key,1)[0]
            assert body==(PAPER/'tables'/f'{name}.tex').read_text()
    definitions=dict(re.findall(r'\\newcommand\{\\(\w+)\}\{([^\n]+)\}',manuscript if inline else macro_text))
    prose=manuscript.split(r'\begin{document}',1)[1]
    source_hashes={}
    for item in ledger:
        assert definitions[item['macro']]==item['display']
        source=ROOT/item['source'];assert source.resolve().is_relative_to(ROOT.resolve()) and source.is_file()
        source_hashes[item['source']]=sha(source)
        count=len(re.findall(r'\\'+item['macro']+r'(?![A-Za-z])',prose))
        item['manuscript_occurrences']=count;item['source_sha256']=source_hashes[item['source']]
        if count:used.append(item)
    # All empirical decimal values in prose must be macros. These literal values
    # are definitions or the recorded noninferiority loss bounds, not observations.
    raw=prose
    raw=re.sub(r'% BEGIN INLINE TABLE [^\n]+\n.*?% END INLINE TABLE [^\n]+','',raw,flags=re.S)
    raw=re.sub(r'% BEGIN INLINE BIBLIOGRAPHY\n.*?% END INLINE BIBLIOGRAPHY','',raw,flags=re.S)
    raw='\n'.join(line for line in raw.splitlines() if not line.startswith((r'\setlength',r'\renewcommand{\arraystretch}')))
    for v in re.findall(r'(?<![\w.])\d+\.\d+(?![\w.])',raw):assert v in ['0.01','0.02','0.5'],v
    stale=['Interim experimental manuscript','Only unaugmented seeds','Seed 123 is training',
           'Augmented-arm comparisons, paired statistical tests, and size ablations remain incomplete',
           'effectiveness of ProvGuard is not yet established','structure-matched hard-negative']
    assert not any(t.lower() in manuscript.lower() for t in stale)
    assert r'\documentclass[conference]{IEEEtran}' in manuscript
    assert 'Anonymous Authors' in manuscript
    assert not any(t in prose for t in [r'\usepackage{geometry}',r'\fontsize{',r'\addtolength{\textwidth}'])
    assert r'\usepackage{geometry}' not in manuscript
    citations={k.strip() for group in re.findall(r'\\cite\{([^}]+)\}',manuscript) for k in group.split(',')}
    bibliography=(ROOT/'ProvGuard.bib').read_text(encoding='utf-8')
    keys=set(re.findall(r'@\w+\{([^,]+),',bibliography));assert keys==citations,(keys,citations)
    bbl_keys=set(re.findall(r'\\bibitem\{([^}]+)\}',manuscript if inline else (PAPER/'ProvGuard.bbl').read_text()));assert bbl_keys==citations
    build=load('paper/build_audit.json')
    assert build['page_count']<=6 and build['overfull_boxes']==0 and build['unresolved_citations']==0 and build['unresolved_references']==0 and build['missing_glyphs']==0
    text=(PAPER/'rendered_text.txt').read_text(encoding='utf-8') if (PAPER/'rendered_text.txt').exists() else '\n'.join(page.get_text() for page in fitz.open(PAPER/'ProvGuard.pdf'))
    assert '??' not in text
    pdf=fitz.open(PAPER/'ProvGuard.pdf')
    fonts=sorted({f[3].split('+')[-1] for page in pdf for f in page.get_fonts()})
    assert any('NimbusRom' in f for f in fonts),'Times-compatible IEEE body font missing'
    for name,entry in load('figures/paper/asset_manifest.json').items():
        for path,h in entry['input_sha256'].items():assert sha(ROOT/path)==h,path
        for ext,h in entry['files'].items():assert sha(ROOT/f'figures/paper/{name}.{ext}')==h
    queue=pd.read_csv(ROOT/'reports/MANUAL_AUDIT_QUEUE.csv',keep_default_na=False)
    assert len(queue)==172 and (queue.mean_attack_probability>=.5).all()
    controls=['provguard_250.csv','provguard_500.csv','provguard_1000.csv','random_real_1000.csv','curated_hn.csv']
    for name in controls:assert sha(ROOT/'data/provguard'/name)==select['augmentation_sha256'][name]
    frozen_master=load('reports/final_source_revision.json')['results_master_sha256']
    assert sha(ROOT/'results_master.csv')==frozen_master
    deep={}
    if args.deep:
        from provguard.study import verify_completed
        from provguard.final_analysis import predictions,rates,align,crossed_bootstrap
        from provguard.leakage import LeakageIndex
        from provguard.core import integrity
        integrity(); print('Frozen benchmark hashes verified.',flush=True)
        for row in registry:
            m,k=row['method'],row['seed'];assert verify_completed(m,k)
            if row.get('model_sha256'):assert sha(ROOT/f'outputs/provguard/{m}/seed_{k}/model/model.safetensors')==row['model_sha256']
            t=rates(predictions(m,k,'test'));hb=predictions(m,k,'hard_benign');h=rates(hb)
            real=rates(hb.loc[hb.source_type=='real']);struct=rates(predictions(m,k,'structural_ood'))
            actual=[t['f1'],t['recall'],h['fpr'],real['fpr'],rates(predictions(m,k,'obfuscated'))['recall'],rates(predictions(m,k,'domain_ood'))['f1'],struct['fpr']]
            saved=master.loc[(master.method==m)&(master.seed==k),METRICS].iloc[0].to_numpy(dtype=float);close(actual,saved)
        deep['primary_runs_verified_against_model_config_and_all_seven_prediction_files']=30
        deep['prediction_derived_primary_metrics_verified']=210
        print('All 30 saved models and 210 paper metrics independently verified.',flush=True)
        # Reproduce all six aggregate intervals directly from paired predictions.
        for _,row in aggregate.iterrows():
            comparator=row.comparison.split(' vs ')[0];d=[]
            for k in SEEDS:
                left=predictions(comparator,k,'hard_benign');right=predictions('provguard_1000',k,'hard_benign')
                if row.subset=='external_hard_benign':left=left.loc[left.source_type=='real'];right=right.loc[right.source_type=='real']
                paired=align(left,right)
                d.append((paired.probability_score_a.to_numpy()>=.5).astype(int)-(paired.probability_score_b.to_numpy()>=.5).astype(int))
            ci=crossed_bootstrap(np.array(d),repetitions=int(row.bootstrap_repetitions),seed=int(row.bootstrap_seed))
            close(ci,[row.bootstrap_ci_lower,row.bootstrap_ci_upper]);close(np.mean(d),row.delta_FPR)
        deep['aggregate_crossed_bootstrap_intervals_reproduced']=6
        index=LeakageIndex.load();checks={}
        for name in controls:
            frame=pd.read_csv(ROOT/'data/provguard'/name,keep_default_na=False)
            checked,report=index.filter(frame,batch=31);assert len(checked)==len(frame) and frame.label.astype(int).eq(0).all()
            checks[name]=report
        deep['independent_persisted_lexical_exclusion']=checks
        deep['no_evaluation_records_added_to_augmentation']=True
        deep['caveat']='Original upstream train is unchanged; lexical exclusion does not certify semantic independence.'
        srcrows=load('reports/secondary_registry.json')['experiments']
        source_completed=[r for r in srcrows if r['family']=='source' and r['status']=='COMPLETED']
        for r in source_completed:assert verify_completed(r['method'],r['seed'])
        deep['source_augmentation_runs_verified']=len(source_completed)
        print('Six aggregate intervals and all augmentation leakage checks reproduced.',flush=True)
    result={'status':'PASSED','required_runs':30,'seeds':SEEDS,'summary_cells_checked':summary_cells,
        'table_sources_checked':6,'ledger_macro_n':len(ledger),'used_prose_macro_n':len(used),
        'used_prose_numeric_occurrences':sum(r['manuscript_occurrences'] for r in used),
        'primary_configuration':1000,'public_hb_n':808,'external_hb_n':208,
        'stale_interim_statements':0,'citation_keys_resolved':len(citations),'fonts':fonts,
        'input_sha256':source_hashes,'tex_sha256':sha(ROOT/'ProvGuard.tex'),'bib_sha256':sha(ROOT/'ProvGuard.bib'),
        'pdf_sha256':sha(PAPER/'ProvGuard.pdf'),'ieee_class_sha256':sha(ROOT/'IEEEtran.cls'),
        'model_results_master_unchanged':True,'deep_verification':deep,'claims':used,
        'single_file_tex':inline,'embedded_tables_checked':3 if inline else 0}
    (PAPER/'numerical_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['claims','input_sha256','fonts','deep_verification']},indent=2))

if __name__=='__main__':main()
