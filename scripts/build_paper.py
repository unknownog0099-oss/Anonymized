"""Compile the unchanged IEEE conference class and render all pages for inspection."""
from pathlib import Path
import json
import os
import re
import shutil
import subprocess
import sys
import pymupdf as fitz

ROOT=Path(__file__).resolve().parents[1]
PAPER=ROOT/'paper'

def main():
    compiler=ROOT/'.cache/latex/tectonic.exe'
    if not compiler.exists():
        found=shutil.which('tectonic')
        if not found:
            raise RuntimeError('Tectonic is required; the original build used .cache/latex/tectonic.exe.')
        compiler=Path(found)
    env=os.environ.copy()
    env['TECTONIC_CACHE_DIR']=str(ROOT/'.cache/latex/bundles')
    command=[str(compiler),'--keep-logs','--keep-intermediates','--outdir','paper','ProvGuard.tex']
    proc=subprocess.run(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
    (PAPER/'compile_console.log').write_text(proc.stdout,encoding='utf-8')
    if proc.returncode:
        print(proc.stdout[-3000:])
        raise RuntimeError(f'LaTeX compilation failed ({proc.returncode}); see paper/compile_console.log.')
    pdf=fitz.open(PAPER/'ProvGuard.pdf')
    preview=PAPER/'preview_final';preview.mkdir(exist_ok=True)
    text=[]
    for i,page in enumerate(pdf):
        page.get_pixmap(matrix=fitz.Matrix(1.7,1.7),alpha=False).save(preview/f'page_{i+1:02d}.png')
        text.append(page.get_text())
    (PAPER/'rendered_text.txt').write_text('\n\f\n'.join(text),encoding='utf-8')
    log=(PAPER/'ProvGuard.log').read_text(errors='replace')
    report={'page_count':len(pdf),'page_size_points':list(pdf[0].rect)[2:],
        'overfull_boxes':len(re.findall(r'Overfull \\[hv]box',log)),
        'underfull_boxes':len(re.findall(r'Underfull \\[hv]box',log)),
        'unresolved_citations':len(re.findall(r'Citation .+ undefined',log)),
        'unresolved_references':len(re.findall(r'Reference .+ undefined',log)),
        'missing_glyphs':len(re.findall(r'Missing character:',log)),
        'bibliography_entries':len(re.findall(r'\\bibitem', (ROOT/'ProvGuard.tex').read_text() if r'\begin{thebibliography}' in (ROOT/'ProvGuard.tex').read_text() else (PAPER/'ProvGuard.bbl').read_text())),
        'compiler':'Tectonic 0.17.0','renderer':'PyMuPDF '+fitz.VersionBind,
        'preview_directory':str(preview.relative_to(ROOT))}
    if report['unresolved_citations'] or report['unresolved_references'] or report['missing_glyphs']:
        raise RuntimeError('Compiled PDF has unresolved references or missing glyphs; inspect paper/ProvGuard.log.')
    (PAPER/'build_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    warnings=[line for line in log.splitlines() if 'Overfull' in line or ('undefined' in line and line.startswith('LaTeX Warning:')) or 'Missing character:' in line]
    if warnings: print('\n'.join(warnings))

if __name__=='__main__':main()
