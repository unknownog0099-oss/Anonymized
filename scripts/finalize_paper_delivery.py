"""Freeze the audited paper deliverables and test a portable LaTeX rebuild."""
from pathlib import Path
import hashlib
import json
import os
import re
import shutil
import subprocess
import zipfile
import pandas as pd
import pymupdf as fitz

ROOT=Path(__file__).resolve().parents[1]
PAPER=ROOT/'paper'
FIG=ROOT/'figures/paper'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def write_audit(portable):
    p=pd.read_csv(ROOT/'reports/PRIMARY_RESULTS.csv').set_index('method')
    c=pd.read_csv(ROOT/'reports/COMPARISON_RESULTS.csv').set_index('comparison')
    s=pd.read_csv(ROOT/'reports/STATISTICAL_RESULTS.csv');s=s.loc[s.seed=='AGGREGATE']
    e=s.loc[(s.comparison=='random_real vs provguard_1000')&(s.subset=='external_hard_benign')].iloc[0]
    mixed=s.loc[(s.comparison=='random_real vs provguard_1000')&(s.subset=='usable_hard_benign')].iloc[0]
    b=load(PAPER/'build_audit.json');n=load(PAPER/'numerical_audit.json')
    assert n['status']=='PASSED' and n['deep_verification']['primary_runs_verified_against_model_config_and_all_seven_prediction_files']==30
    assert n['tex_sha256']==sha(ROOT/'ProvGuard.tex') and n['pdf_sha256']==sha(PAPER/'ProvGuard.pdf') and n['bib_sha256']==sha(ROOT/'ProvGuard.bib')
    base=c.loc['baseline vs provguard_1000'];plot_names=[f.stem for f in FIG.glob('*.pdf')]
    text=f'''# Final paper audit

Final title: **ProvGuard: Confusion-Selected Hard Negatives for Reducing Prompt-Injection Over-Defense under Distribution Shift**.

The anonymous manuscript is a completed empirical study of the required experiment matrix. It retains mixed outcomes and explicitly separates incomplete optional checks. No acceptance or universal-superiority claim is made.

## Template and presentation

- Original `IEEEtran` conference class, version 1.8b, from the supplied template; class SHA-256 `{sha(ROOT/'IEEEtran.cls')}`.
- Final length: **{b['page_count']} US-letter pages, including references**. Body remains the template's 10-point Times-family type (rendered as Nimbus Roman); no margin, geometry, or body-font reduction.
- Anonymous IEEE author block retained; no funding, affiliation, or conference identity invented.
- All six final rendered pages and seven figure assets were visually inspected after iterative compilation. No clipped tables, overlapping labels, missing glyphs, or orphan headings remain. The final bibliography columns are balanced using the class's own reference-trigger command.
- Final log: **{b['overfull_boxes']} overfull boxes, {b['unresolved_citations']} unresolved citations, {b['unresolved_references']} unresolved cross-references, {b['missing_glyphs']} missing-glyph warnings**. One ordinary underfull vertical-box warning remains during IEEE float packing; inspected columns are filled and unclipped. Initial XeTeX class font-encoding warnings resolve to the verified Times-compatible body fonts.

## Completed experimental evidence

- **30/30 required runs**, six arms, five training seeds **13, 42, 123, 2024, 7777**.
- All thirty saved model/configuration bindings and all seven prediction files per run checked independently; **210 paper metrics** recalculated from probabilities and compared with the frozen master results.
- Required arms: Baseline, Curated-HN, Random-Real-1000, ProvGuard-250/500/1000.
- Curated augmentation: **343 valid unique rows**; no size-matching claim. Both primary real augmentation arms have 1000 rows.
- Six completed three-seed source-augmentation checks were verified and reported descriptively. They change augmentation source only; original PIDS training is retained. No fully held-out-corpus claim.
- Optional DeBERTa baseline attempt failed because its pretrained assets could not be downloaded or found in cache. Remaining architecture runs are not reported as completed. Extra random-control samples await semantic reviews; no result is fabricated.

## Correct denominators and selection

- Released HB rows: **1472**. Accessible evaluation text: **808 = 600 curated + 208 external**. **664 withheld LMSYS texts excluded**.
- External corpus coverage: 68 Dolly and 140 OASST1 prompts. Stress-set FPR is not a deployment prevalence estimate.
- Original PIDS train/validation/IID counts: 27093 / 3989 / 3918.
- Primary configuration is predefined **K=1000**, not the test-best K. K=250 and K=500 are nested ablations; all thresholds are fixed at 0.5.
- Selector ensemble: baseline seeds 13/42/123, fixed for the study. Leakage-filtered real pool: 17400. Two separately instantiated automated passes review 2927 records; both BENIGN is required. This is **not human annotation** and shared model errors remain possible.
- Source/length quotas match between the two real primary arms; sample overlap of 213 examples is permitted. No syntactic or structure-matching claim.

## Headline numerical checks

| Quantity | Final measured value |
|---|---:|
| Baseline external FPR | {100*p.loc['baseline','external_hb_fpr_mean']:.2f}% |
| Curated-HN external FPR | {100*p.loc['curated','external_hb_fpr_mean']:.2f}% |
| Random-Real external FPR | {100*p.loc['random_real','external_hb_fpr_mean']:.2f}% |
| ProvGuard external FPR | {100*p.loc['provguard_1000','external_hb_fpr_mean']:.2f}% |
| Baseline-to-ProvGuard external reduction | {100*base.external_hb_fpr_absolute_reduction:.2f} pp; {100*base.external_hb_fpr_relative_reduction:.1f}% relative |
| IID F1, baseline to ProvGuard | {100*p.loc['baseline','iid_f1_mean']:.2f}% to {100*p.loc['provguard_1000','iid_f1_mean']:.2f}% |
| IID F1 change, from unrounded means | {100*base.iid_f1_provguard_minus_comparator:+.2f} pp |
| Attack recall, baseline to ProvGuard | {100*p.loc['baseline','iid_recall_mean']:.2f}% to {100*p.loc['provguard_1000','iid_recall_mean']:.2f}% |
| Attack recall change, from unrounded means | {100*base.iid_recall_provguard_minus_comparator:+.2f} pp |
| External Random-Real minus ProvGuard FPR CI | [{e.bootstrap_ci_lower:.4f}, {e.bootstrap_ci_upper:.4f}] |
| Aggregate Random-Real minus ProvGuard FPR CI | [{mixed.bootstrap_ci_lower:.4f}, {mixed.bootstrap_ci_upper:.4f}] |

The external Random-Real comparison is supported by a positive **pointwise**, unadjusted crossed-bootstrap interval. The aggregate interval crosses zero; the predefined joint novelty verdict remains **inconclusive**. Curated-HN wins aggregate HB FPR. Structural benign FPR decreases, but domain-OOD F1 also decreases. All six aggregate 10000-resample intervals were reproduced from paired seed/shared-prompt predictions. Seed predictions are not pooled as independent examples.

## Data separation and validity limits

No accessible evaluation record was added as training augmentation. The original frozen train and validation files are unchanged. A freshly loaded persisted TF-IDF character-boundary 3--5 index rejects exact/high-lexical-similarity matches at cosine 0.90; all five augmentation files independently pass. The unsupervised exclusion representation is fitted on available benchmark and potential augmentation text; this exposure is disclosed. It does not guarantee semantic independence or certify original upstream splits beyond their frozen hashes.

The candidate/evaluation corpora share provenance; externally sourced is not equivalent to unseen corpus. Automated adjudication is fallible, scores were hidden but ranked order was visible, and the exact serving model version was not exposed. The **172-row** `reports/MANUAL_AUDIT_QUEUE.csv` provides high-score texts for a future human review. Its human fields are empty; no completed human audit is claimed.

## Numerical, citation, and reviewer audits

- `{n['summary_cells_checked']}` summary mean/SD cells checked against per-seed results; all **six table sources** checked in exact numeric order.
- `{n['used_prose_macro_n']}` used numerical macros bind `{n['used_prose_numeric_occurrences']}` prose/caption occurrences to canonical evidence. The ledger records calculation, display rounding, source path, and input hash. Remaining literal decimal values are definitions or recorded protocol bounds, not copied observations.
- All **14** bibliography entries are cited, checked against primary metadata, and resolve in the compiled bibliography. No unused template references or `??` markers.
- Source metadata and DOI/arXiv identifiers are documented in `paper/citation_audit.json`; some publisher resolvers were inaccessible, so successful live resolution of every DOI is not asserted.
- No stale partial-result statements remain in the final manuscript. Optional unfinished checks are explicitly distinguished from the completed required study.
- Figures and tables use final artifacts only, with color/pattern/marker consistency and numeric input hashes. Seven vector PDFs, seven editable SVGs, seven 400 dpi PNGs, and the shared Python generation source are supplied.
- `paper/REVIEWER_PASS.md` records the fifteen requested skeptical objections and the manuscript responses. Literature positioning acknowledges established mining and prior over-defense mitigation.

## Portable build and files

Portable package rebuild: **{portable}**. The archive includes every local file referenced by the LaTeX manuscript. Standard LaTeX recompilation requires no experiment data; figure regeneration and deep evidence checks require the original workspace.

Final assets: `ProvGuard.tex`, `ProvGuard.bib`, `paper/ProvGuard.pdf`, `figures/paper/`, `paper/tables/`, the three build/audit scripts, claim and citation JSON, and this audit. Historical manuscript artifacts are isolated in `paper/archive_interim/` and are not included in the submission package.

Final source SHA-256: `{sha(ROOT/'ProvGuard.tex')}`.
Final bibliography SHA-256: `{sha(ROOT/'ProvGuard.bib')}`.
Final PDF SHA-256: `{sha(PAPER/'ProvGuard.pdf')}`.
Frozen master results SHA-256: `{sha(ROOT/'results_master.csv')}` (unchanged).
'''
    (ROOT/'PAPER_AUDIT.md').write_text(text,encoding='utf-8');(PAPER/'PAPER_AUDIT.md').write_text(text,encoding='utf-8')
    b['rendered_pages_visually_inspected']=list(range(1,7));b['figure_assets_visually_inspected']=sorted(plot_names)
    b['final_tex_sha256']=sha(ROOT/'ProvGuard.tex');b['final_pdf_sha256']=sha(PAPER/'ProvGuard.pdf')
    b['body_font_verified']='Times-compatible Nimbus Roman; IEEE 10-point body'
    b['portable_package_build']=portable
    (PAPER/'build_audit.json').write_text(json.dumps(b,indent=2),encoding='utf-8')

def package():
    files=[ROOT/p for p in ['ProvGuard.tex','ProvGuard.bib','IEEEtran.cls','IEEEtran.bst','PAPER_AUDIT.md',
        'paper/ProvGuard.pdf','paper/ProvGuard.bbl','paper/results_macros.tex','paper/results_snapshot.csv',
        'paper/numerical_claims.json','paper/numerical_audit.json','paper/citation_audit.json','paper/build_audit.json',
        'paper/REVIEWER_PASS.md','paper/DELIVERY_README.md','scripts/build_paper_assets.py','scripts/build_paper.py',
        'scripts/audit_paper.py','scripts/finalize_paper_delivery.py']]
    files+=sorted(FIG.glob('*.pdf'))+sorted(FIG.glob('*.svg'))+sorted(FIG.glob('*.png'))+[FIG/'asset_manifest.json']
    files+=sorted((PAPER/'tables').glob('*.tex'))
    if (PAPER/'portable_build_audit.json').exists():files.append(PAPER/'portable_build_audit.json')
    files+=sorted((PAPER/'preview_final').glob('page_*.png'))
    target=ROOT/'ProvGuard_final_submission.zip'
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for f in files:
            assert f.is_file() and f.resolve().is_relative_to(ROOT.resolve())
            z.write(f,f.relative_to(ROOT).as_posix())
        z.writestr('README.md',(PAPER/'DELIVERY_README.md').read_text(encoding='utf-8'))
    return target

def finalize_single():
    proof=load(PAPER/'single_tex_audit.json');numeric=load(PAPER/'numerical_audit.json')
    assert proof['status']=='PASSED' and proof['single_tex_sha256']==sha(ROOT/'ProvGuard.tex')
    assert proof['final_pdf_sha256']==sha(PAPER/'ProvGuard.pdf')
    assert numeric['status']=='PASSED' and numeric['tex_sha256']==sha(ROOT/'ProvGuard.tex')
    archive=ROOT/'ProvGuard_final_submission.zip';backup=PAPER/'ProvGuard_modular_submission.zip'
    if archive.exists() and not backup.exists():shutil.copyfile(archive,backup)
    old=PAPER/'PAPER_AUDIT_modular.md'
    if not old.exists():shutil.copyfile(ROOT/'PAPER_AUDIT.md',old)
    text=old.read_text(encoding='utf-8').split('## Portable build and files',1)[0]
    text+='''## Single-file build and files

The final `ProvGuard.tex` embeds all 153 numerical macro definitions, the three
manuscript tables, fourteen IEEE-formatted references, and all seven vector
figures as native TikZ geometry. It contains no external project-file calls.
The manuscript uses four embedded figures; the other three remain available
as optional macros. Standard IEEEtran and declared TeX packages are required.

**PASSED:** a fresh folder containing only `ProvGuard.tex` compiled to six pages
with identical extracted text, resolved citations/references, and zero overfull
boxes or missing glyphs. All six final pages were inspected, including corrected
PDF hatch tiling. The current numeric audit checks embedded macros and tables
against the same canonical experiment evidence. The earlier independent deep
model/prediction audit remains archived in `paper/numerical_audit_modular.json`;
models and experiments were not rerun for the single-file conversion.

The portable archive contains the single source, compiled PDF, build and
numerical audits, and compilation instructions. Previous modular source, PDF,
audit, and archive are preserved under `paper/`.

'''
    text+=f"Final source SHA-256: `{sha(ROOT/'ProvGuard.tex')}`.\n\nFinal PDF SHA-256: `{sha(PAPER/'ProvGuard.pdf')}`.\n\nFrozen master results SHA-256: `{sha(ROOT/'results_master.csv')}` (unchanged).\n"
    for path in [ROOT/'PAPER_AUDIT.md',PAPER/'PAPER_AUDIT.md']:path.write_text(text,encoding='utf-8')
    build=load(PAPER/'build_audit.json')
    build.update(rendered_pages_visually_inspected=list(range(1,7)),final_tex_sha256=sha(ROOT/'ProvGuard.tex'),
        final_pdf_sha256=sha(PAPER/'ProvGuard.pdf'),portable_single_tex_build='PASSED')
    (PAPER/'build_audit.json').write_text(json.dumps(build,indent=2),encoding='utf-8')
    files=['ProvGuard.tex','paper/ProvGuard.pdf','PAPER_AUDIT.md','paper/single_tex_audit.json',
           'paper/numerical_audit.json','paper/build_audit.json','paper/DELIVERY_README.md']
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name in files:z.write(ROOT/name,name)
        z.writestr('README.md',(PAPER/'DELIVERY_README.md').read_text(encoding='utf-8'))
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    print(json.dumps({'archive':str(archive),'file_n':len(files)+1,'single_tex_build':'PASSED','paper_pages':6},indent=2))

def main():
    if '% BEGIN SELF-CONTAINED VECTOR ARTWORK' in (ROOT/'ProvGuard.tex').read_text(encoding='utf-8'):
        finalize_single();return
    write_audit('testing')
    archive=package();dest=ROOT/'tmp/pdfs/package_check';dest.mkdir(parents=True,exist_ok=True)
    assert dest.resolve().is_relative_to((ROOT/'tmp').resolve())
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():assert (dest/info.filename).resolve().is_relative_to(dest.resolve())
        z.extractall(dest)
    env=os.environ.copy();env['TECTONIC_CACHE_DIR']=str(ROOT/'.cache/latex/bundles')
    result=subprocess.run([str(ROOT/'.cache/latex/tectonic.exe'),'--only-cached','--keep-logs','--keep-intermediates','--outdir','paper','ProvGuard.tex'],
        cwd=dest,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
    (PAPER/'portable_compile_console.log').write_text(result.stdout,encoding='utf-8')
    assert result.returncode==0,'Portable compilation failed; see paper/portable_compile_console.log.'
    portable=fitz.open(dest/'paper/ProvGuard.pdf');original=fitz.open(PAPER/'ProvGuard.pdf')
    assert len(portable)==len(original)==6
    assert [p.get_text() for p in portable]==[p.get_text() for p in original],'Portable paper text differs.'
    log=(dest/'paper/ProvGuard.log').read_text(errors='replace')
    assert not re.search(r'Overfull \\[hv]box|Citation .+ undefined|Reference .+ undefined|Missing character:',log)
    audit={'status':'PASSED','page_count':6,'extracted_pdf_text_identical':True,
        'template_class_sha256':sha(dest/'IEEEtran.cls'),'tex_sha256':sha(dest/'ProvGuard.tex'),
        'bib_sha256':sha(dest/'ProvGuard.bib'),'all_manuscript_assets_present':True}
    (PAPER/'portable_build_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    write_audit('PASSED: extracted archive recompiled to six pages with identical extracted text and resolved references')
    archive=package()
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        print(json.dumps({'archive':str(archive),'file_n':len(z.namelist()),'size_bytes':archive.stat().st_size,
            'portable_build':'PASSED','paper_pages':6,'numeric_audit':'PASSED'},indent=2))

if __name__=='__main__':main()
