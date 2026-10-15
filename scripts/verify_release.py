"""Read-only checks of the anonymous source/evidence release; no model loading."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import sys
import numpy as np
import pandas as pd
import pymupdf as fitz

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
METHODS=['baseline','curated','random_real','provguard_250','provguard_500','provguard_1000']
SEEDS=[13,42,123,2024,7777]
METRICS=['iid_f1','iid_recall','hard_benign_fpr','external_hb_fpr','obfuscated_recall','domain_ood_f1','structural_ood_fpr']
SECRET=re.compile(r'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|hf_[A-Za-z0-9]{25,}|sk-[A-Za-z0-9_-]{30,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)')
TEXT={'.py','.ps1','.md','.json','.jsonl','.csv','.tex','.bib','.toml','.txt','.svg','.bst','.cls','.cff'}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads((ROOT/p).read_text(encoding='utf-8'))
def close(a,b):assert np.allclose(a,b,rtol=0,atol=2e-14),(a,b)

def files():
    for base,dirs,names in os.walk(ROOT):
        dirs[:]=[d for d in dirs if d not in {'.git','.cache','.venv','__pycache__','.pytest_cache','tmp','outputs','models','work'}]
        for name in names:yield Path(base)/name

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source-workspace',type=Path)
    args=parser.parse_args();manifest=load('artifacts/release_manifest.json')
    for name,record in manifest['files'].items():
        path=ROOT/name;assert path.resolve().is_relative_to(ROOT.resolve()) and path.is_file(),name
        assert sha(path)==record['sha256'],f'Release checksum changed: {name}'
        assert path.stat().st_size<100*2**20,f'File exceeds ordinary GitHub limit: {name}'
    print('Release file hashes and GitHub file sizes verified.',flush=True)
    markers=[]
    if args.source_workspace:
        original=args.source_workspace.resolve()
        markers=[str(original),original.as_posix(),str(original).replace('\\','\\\\'),original.name]
        # Known machine username, distinct from public dataset authors/strings.
        for i,p in enumerate(original.parts):
            if p.lower()=='users' and i+1<len(original.parts):
                profile=Path(*original.parts[:i+2]);markers.extend([str(profile),profile.as_posix(),str(profile).replace('\\','\\\\')])
    checked=0;violations=[];metadata_n=0
    for path in files():
        rel=path.relative_to(ROOT).as_posix()
        if path.name in {'.env','.git','credentials.json'} or path.suffix in {'.pem','.key','.safetensors','.pt','.exe'}:
            violations.append({'file':rel,'kind':'excluded/private file'})
        if path.suffix=='.pdf':
            with fitz.open(path) as doc:
                metadata_n+=1
                assert not doc.metadata.get('author') and not doc.metadata.get('creationDate') and not doc.metadata.get('modDate'),rel
                assert not doc.xref_xml_metadata(),rel
                text='\n'.join(p.get_text() for p in doc)
        elif path.suffix in TEXT or path.name in {'LICENSE','.gitignore','.gitattributes'}:
            text=path.read_text(encoding='utf-8-sig');checked+=1
        else:
            # Check canonical binary artifacts for embedded original local paths.
            raw=path.read_bytes()
            if any(m.encode() in raw for m in markers if ':' in m or '/' in m):violations.append({'file':rel,'kind':'embedded local profile/project path'})
            continue
        if any(m.lower() in text.lower() for m in markers):violations.append({'file':rel,'kind':'original workspace/profile marker'})
        if SECRET.search(text):violations.append({'file':rel,'kind':'credential/private-key pattern'})
    assert not violations,json.dumps(violations,indent=2)
    print('Text, PDF metadata, and known local-identity/credential patterns checked.',flush=True)
    from provguard.core import SETS
    bench=ROOT/'upstream/data/pids_bench_v3';freeze=(bench/'FREEZE_MANIFEST.txt').read_text()
    for relative in SETS.values():
        path=bench/relative;match=re.search(rf'^{path.stem}\s+\d+\s+\d+\s+\d+\s+([a-f0-9]{{64}})',freeze,re.M)
        assert match and sha(path)==match[1],relative
    from provguard.leakage import LeakageIndex
    index=LeakageIndex.load();selection=load('reports/candidate_selection_audit.json')
    for name,expected in selection['augmentation_sha256'].items():assert sha(ROOT/'data/provguard'/name)==expected
    assert sha(ROOT/'data/provguard/semantic_review.csv')==selection['semantic_review_sha256']
    for name in ['provguard_250','provguard_500','provguard_1000','random_real_1000','curated_hn']:
        frame=pd.read_csv(ROOT/f'data/provguard/{name}.csv',keep_default_na=False)
        assert frame.label.astype(int).eq(0).all()
    master=pd.read_csv(ROOT/'results_master.csv');assert len(master)==30
    assert set(zip(master.method,master.seed))=={(m,s) for m in METHODS for s in SEEDS}
    assert master.threshold.eq(.5).all()
    from provguard.final_analysis import rates,align,crossed_bootstrap
    artifact=ROOT/'artifacts/experiments';metric_n=0
    def pred(m,s,subset):return pd.read_csv(artifact/m/f'seed_{s}'/f'{subset}_predictions.csv',keep_default_na=False)
    for m,s in [(m,s) for m in METHODS for s in SEEDS]:
        base=artifact/m/f'seed_{s}';config=json.loads((base/'config.json').read_text())
        assert config['threshold']==.5 and config['seed']==s and config['method']==m
        assert config['train_sha256']==sha(bench/'train.csv') and config['validation_sha256']==sha(bench/'val.csv')
        for subset,relative in SETS.items():
            if subset=='train':continue
            actual=pred(m,s,subset);reference=pd.read_csv(bench/relative,keep_default_na=False)
            valid=reference.text.map(lambda x:isinstance(x,str) and bool(x.strip()) and x.strip().lower() not in {'nan','null'})
            reference=reference.loc[valid].copy();reference['eval_id']=[f'{Path(relative).stem}:{i}' for i in reference.index]
            merged=reference[['eval_id','text','label']].merge(actual,on='eval_id',suffixes=('_frozen',''),validate='one_to_one')
            assert len(actual)==len(reference)==len(merged) and actual.eval_id.is_unique,(m,s,subset)
            assert merged.text_frozen.eq(merged.text).all() and merged.label.astype(int).eq(merged.true_label.astype(int)).all()
            probabilities=actual.probability_score.to_numpy(float)
            assert np.isfinite(probabilities).all() and ((probabilities>=0)&(probabilities<=1)).all()
            assert np.array_equal((probabilities>=.5).astype(int),actual.predicted_label.astype(int))
        test=rates(pred(m,s,'test'));hb=pred(m,s,'hard_benign');struct=rates(pred(m,s,'structural_ood'))
        actual=[test['f1'],test['recall'],rates(hb)['fpr'],rates(hb.loc[hb.source_type.eq('real')])['fpr'],
            rates(pred(m,s,'obfuscated'))['recall'],rates(pred(m,s,'domain_ood'))['f1'],struct['fpr']]
        close(actual,master.loc[(master.method==m)&(master.seed==s),METRICS].iloc[0].to_numpy(float));metric_n+=7
    print('All 30 primary runs: frozen row/text/label bindings and 210 metrics verified.',flush=True)
    stats=pd.read_csv(ROOT/'reports/STATISTICAL_RESULTS.csv');aggregate=stats.loc[stats.seed.eq('AGGREGATE')]
    assert len(aggregate)==6
    for _,row in aggregate.iterrows():
        comparator=row.comparison.split(' vs ')[0];delta=[]
        for s in SEEDS:
            a=pred(comparator,s,'hard_benign');b=pred('provguard_1000',s,'hard_benign')
            if row.subset=='external_hard_benign':a=a.loc[a.source_type.eq('real')];b=b.loc[b.source_type.eq('real')]
            paired=align(a,b);delta.append((paired.probability_score_a.to_numpy()>=.5).astype(int)-(paired.probability_score_b.to_numpy()>=.5).astype(int))
        close(crossed_bootstrap(np.array(delta),repetitions=int(row.bootstrap_repetitions),seed=int(row.bootstrap_seed)),[row.bootstrap_ci_lower,row.bootstrap_ci_upper])
        close(np.mean(delta),row.delta_FPR)
    source=pd.read_csv(ROOT/'reports/SECONDARY_RESULTS.csv');assert len(source)==6
    for _,row in source.iterrows():
        hb=pred(row.method,int(row.seed),'hard_benign')
        close(rates(hb.loc[hb.source_type.eq('real')])['fpr'],row.external_hb_fpr)
    with fitz.open(ROOT/'paper/ProvGuard.pdf') as doc:assert len(doc)==6
    tex=(ROOT/'ProvGuard.tex').read_text(encoding='utf-8')
    assert 'Anonymous Authors' in tex and not re.search(r'\\(?:input|include|includegraphics|bibliography|bibliographystyle)\s*[\[{]',tex)
    result={'status':'PASSED','manifest_files_checked':len(manifest['files']),'text_files_scanned':checked,
        'pdf_metadata_checked':metadata_n,'known_local_identity_matches':0,'credential_pattern_matches':0,
        'primary_prediction_sets_checked':210,'prediction_derived_metrics_checked':metric_n,
        'aggregate_crossed_bootstrap_intervals_reproduced':6,'completed_source_runs_checked':6,
        'frozen_benchmark_files_checked':8,'canonical_leakage_representation_hashes_checked':True,
        'five_primary_augmentation_hashes_checked':True,'paper_pages':6,
        'model_weights_verified':False,'anonymity_scope':'Files and known local metadata; account/remote/commit identity is controlled at upload.'}
    (ROOT/'artifacts/RELEASE_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
