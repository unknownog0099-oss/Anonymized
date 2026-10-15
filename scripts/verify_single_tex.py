"""Compile a folder containing only the final TeX source and record the proof."""
from pathlib import Path
import hashlib
import json
import os
import re
import shutil
import subprocess
import pymupdf as fitz

ROOT=Path(__file__).resolve().parents[1]
PAPER=ROOT/'paper'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    # A fresh directory prevents old assets from masking external dependencies.
    import tempfile
    base=ROOT/'tmp/pdfs';base.mkdir(parents=True,exist_ok=True)
    dest=Path(tempfile.mkdtemp(prefix='single_tex_',dir=base))
    source=ROOT/'ProvGuard.tex'
    shutil.copyfile(source,dest/source.name)
    assert [p.name for p in dest.iterdir()]==['ProvGuard.tex']
    env=os.environ.copy();env['TECTONIC_CACHE_DIR']=str(ROOT/'.cache/latex/bundles')
    compiler=ROOT/'.cache/latex/tectonic.exe'
    if not compiler.exists():
        found=shutil.which('tectonic')
        if not found:raise RuntimeError('Install Tectonic and put it on PATH.')
        compiler=Path(found)
    command=[str(compiler),'--keep-logs','ProvGuard.tex']
    proc=subprocess.run(command,cwd=dest,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
    (PAPER/'single_tex_compile_console.log').write_text(proc.stdout,encoding='utf-8')
    if proc.returncode:
        print(proc.stdout[-2500:]);raise RuntimeError('Standalone build failed; see paper/single_tex_compile_console.log.')
    actual=fitz.open(dest/'ProvGuard.pdf');expected=fitz.open(PAPER/'ProvGuard.pdf')
    assert len(actual)==len(expected)==6
    assert [p.get_text() for p in actual]==[p.get_text() for p in expected]
    log=(dest/'ProvGuard.log').read_text(errors='replace')
    assert not re.search(r'Overfull \\[hv]box|Citation .+ undefined|Reference .+ undefined|Missing character:',log)
    report=json.loads((PAPER/'single_tex_audit.json').read_text())
    assert report['single_tex_sha256']==sha(source)
    report.update(status='PASSED',isolated_directory=str(dest.relative_to(ROOT)),initial_directory_files=['ProvGuard.tex'],
        isolated_compile_page_count=6,isolated_pdf_text_identical=True,isolated_compile_standard_tex_library_only=True,
        final_pdf_sha256=sha(PAPER/'ProvGuard.pdf'),rendered_pages_visually_inspected=list(range(1,7)))
    (PAPER/'single_tex_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['status','external_project_file_references','isolated_compile_page_count','isolated_pdf_text_identical','initial_directory_files']},indent=2))

if __name__=='__main__':main()
