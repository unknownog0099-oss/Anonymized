"""Freeze the audited paper as one TeX file with native inline vector artwork.

Standard IEEEtran, PGF/TikZ, and Helvetica packages remain TeX-library dependencies.
No project data, image file, table input, or bibliography file is needed to compile.
The original modular source and PDF are preserved before conversion.
"""
from pathlib import Path
import collections
import hashlib
import json
import math
import re
import shutil
import pymupdf as fitz

ROOT=Path(__file__).resolve().parents[1]
PAPER=ROOT/'paper'
NAMES={'methodology':'PGMethodology','external_hb':'PGExternalHB','ablation':'PGAblation',
       'hardness_distribution':'PGHardness','primary_results':'PGPrimaryResults',
       'security_usability':'PGSecurityUsability','graphical_abstract':'PGGraphicalAbstract'}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def f(v):return f'{float(v):.4f}'.rstrip('0').rstrip('.') or '0'
def pt(p):return '('+f(p[0])+','+f(p[1])+')'

def tex_text(text):
    out=[]
    special={'\\':r'\textbackslash{}','{':r'\{','}':r'\}','$':r'\$','&':r'\&','#':r'\#','%':r'\%','_':r'\_',
             '^':r'\textasciicircum{}','~':r'\textasciitilde{}','−':r'$-$','≥':r'$\geq$','≤':r'$\leq$',
             '±':r'$\pm$','→':r'$\rightarrow$','↑':r'$\uparrow$','↓':r'$\downarrow$','–':'--','—':'---'}
    for c in text:out.append(special.get(c,c))
    return ''.join(out)

def path_code(items,close=False):
    pieces=[];last=None
    for item in items:
        kind=item[0]
        if kind=='l':
            p,q=item[1:];
            if last is None or abs(last.x-p.x)>1e-4 or abs(last.y-p.y)>1e-4:pieces.append(pt(p))
            pieces.append('-- '+pt(q));last=q
        elif kind=='c':
            p,c1,c2,q=item[1:]
            if last is None or abs(last.x-p.x)>1e-4 or abs(last.y-p.y)>1e-4:pieces.append(pt(p))
            pieces.append('.. controls '+pt(c1)+' and '+pt(c2)+' .. '+pt(q));last=q
        elif kind=='re':
            r=item[1];points=[(r.x0,r.y0),(r.x1,r.y0),(r.x1,r.y1),(r.x0,r.y1)]
            if len(item)>2 and item[2]<0:points=points[::-1]
            pieces.append(' -- '.join(pt(p) for p in points)+' -- cycle');last=None
        elif kind=='qu':
            q=item[1];pieces.append(' -- '.join(pt(p) for p in [q.ul,q.ur,q.lr,q.ll])+' -- cycle');last=None
        else:raise ValueError('Unsupported vector primitive '+kind)
    if close and last is not None:pieces.append('-- cycle')
    return ' '.join(pieces)

def color(value,colors):
    value=(0.,0.,0.) if value is None else tuple(float(v) for v in value)
    if len(value)!=3:raise ValueError('Non-RGB artwork color')
    key=tuple(round(v,6) for v in value)
    if key not in colors:colors[key]='PGColor'+str(len(colors))
    return colors[key]

def drawing_options(d,colors):
    options=[];kind=d['type']
    if 'f' in kind:
        options.append('fill='+color(d.get('fill'),colors))
        if d.get('fill_opacity',1)<1:options.append('fill opacity='+f(d['fill_opacity']))
        if d.get('even_odd'):options.append('even odd rule')
    else:options.append('fill=none')
    if 's' in kind:
        options+=['draw='+color(d.get('color'),colors),'line width='+f(d.get('width') or .1)+'bp']
        cap=d.get('lineCap');cap=max(cap) if isinstance(cap,tuple) else cap
        options.append('line cap='+{0:'butt',1:'round',2:'rect'}.get(cap,'butt'))
        options.append('line join='+{0:'miter',1:'round',2:'bevel'}.get(d.get('lineJoin'),'round'))
        if d.get('stroke_opacity',1)<1:options.append('draw opacity='+f(d['stroke_opacity']))
        dash=d.get('dashes') or '[] 0';m=re.match(r'\[([^]]*)\]\s*([\d.-]+)',dash)
        if m and m.group(1).strip():
            vals=[float(v) for v in m.group(1).split()]
            options.append('dash pattern='+' '.join(('on ' if i%2==0 else 'off ')+f(v)+'bp' for i,v in enumerate(vals)))
            options.append('dash phase='+m.group(2)+'bp')
    else:options.append('draw=none')
    return ','.join(options)

def vector_macro(pdf_file,name,colors):
    doc=fitz.open(pdf_file);page=doc[0];w,h=page.rect.width,page.rect.height
    lines=[f'% BEGIN INLINE VECTOR {name}; audited source PDF SHA-256 {sha(pdf_file)}',
           '\\newcommand{\\'+name+r'}[1]{%',r'\resizebox{#1}{!}{%',
           r'\begin{tikzpicture}[x=1bp,y=-1bp]',
           r'\path[use as bounding box] (0,0) rectangle ('+f(w)+','+f(h)+');',
           r'\begin{pgfinterruptboundingbox}']
    scopes=[];primitive_n=0;pattern_n=0
    drawings=page.get_drawings(extended=True);skip_until=0
    for index,d in enumerate(drawings):
        if index<skip_until:continue
        level=d.get('level',0)
        while scopes and scopes[-1]>=level:lines.append(r'\end{scope}');scopes.pop()
        kind=d['type']
        if kind=='clip':
            lines.append(r'\begin{scope}')
            lines.append(r'\clip'+('[even odd rule]' if d.get('even_odd') else '')+' '+path_code(d['items'],d.get('closePath'))+';')
            scopes.append(level)
            # MuPDF exposes only representative tiles for PDF hatch patterns.
            # Repeat the original tile geometry over the entire clipping path.
            children=[];end=index+1
            while end<len(drawings) and drawings[end].get('level',0)>level:
                children.append(drawings[end]);end+=1
            if len(children)>=2 and children[0]['type']=='f' and children[1]['type']=='fs':
                base=children[0]['rect']
                if abs(base.width-72)<.01 and abs(base.height-72)<.01:
                    clip=d['scissor'];pattern_n+=1
                    for ix in range(math.floor((clip.x0-base.x0)/72),math.ceil((clip.x1-base.x0)/72)):
                        for iy in range(math.floor((clip.y0-base.y0)/72),math.ceil((clip.y1-base.y0)/72)):
                            delta=fitz.Point(ix*72,iy*72)
                            for tile in children[:2]:
                                moved=[]
                                for item in tile['items']:
                                    if item[0] in ('l','c'):moved.append((item[0],*[p+delta for p in item[1:]]))
                                    elif item[0]=='re':moved.append(('re',item[1]+(delta.x,delta.y,delta.x,delta.y),*item[2:]))
                                    else:raise ValueError('Unsupported hatch primitive '+item[0])
                                lines.append(r'\path['+drawing_options(tile,colors)+'] '+path_code(moved,tile.get('closePath'))+';')
                                primitive_n+=len(moved)
                    skip_until=end
        elif kind=='group':
            opts=[]
            if d.get('opacity',1)<1:opts.append('opacity='+f(d['opacity']))
            lines.append(r'\begin{scope}'+('['+','.join(opts)+']' if opts else ''));scopes.append(level)
        elif kind in ['f','s','fs']:
            lines.append(r'\path['+drawing_options(d,colors)+'] '+path_code(d['items'],d.get('closePath'))+';');primitive_n+=len(d['items'])
        else:raise ValueError('Unsupported drawing type '+kind)
    while scopes:lines.append(r'\end{scope}');scopes.pop()
    text_n=0
    for block in page.get_text('dict')['blocks']:
        for line in block.get('lines',[]):
            dx,dy=line.get('dir',(1,0));angle=math.degrees(math.atan2(-dy,dx))
            for span in line['spans']:
                if not span['text'].strip():continue
                text_n+=1;c=span.get('color',0);rgb=((c>>16&255)/255,(c>>8&255)/255,(c&255)/255)
                font=r'\fontsize{'+f(span['size'])+'bp}{'+f(span['size']*1.2)+r'bp}\selectfont\sffamily'
                if 'bold' in span['font'].lower():font+=r'\bfseries'
                if 'italic' in span['font'].lower() or 'oblique' in span['font'].lower():font+=r'\itshape'
                opts='anchor=base west,inner sep=0pt,outer sep=0pt,text='+color(rgb,colors)+',font={'+font+'}'
                if abs(angle)>.01:opts+=',rotate='+f(angle)
                lines.append(r'\node['+opts+'] at '+pt(span['origin'])+' {'+tex_text(span['text'])+'};')
    lines += [r'\end{pgfinterruptboundingbox}',r'\end{tikzpicture}%',r'}%',r'}',f'% END INLINE VECTOR {name}']
    doc.close()
    return '\n'.join(lines),{'pdf_sha256':sha(pdf_file),'vector_primitive_n':primitive_n,'text_span_n':text_n,'width_bp':w,'height_bp':h,'retiled_hatch_regions':pattern_n}

def main():
    source_path=ROOT/'ProvGuard.tex';modular=PAPER/'ProvGuard_modular.tex'
    if not modular.exists():shutil.copyfile(source_path,modular)
    if not (PAPER/'ProvGuard_modular.pdf').exists():shutil.copyfile(PAPER/'ProvGuard.pdf',PAPER/'ProvGuard_modular.pdf')
    source=modular.read_text(encoding='utf-8');macros=(PAPER/'results_macros.tex').read_text()
    source=source.replace(r'\input{paper/results_macros}', '% BEGIN INLINE NUMERICAL MACROS\n'+macros+'% END INLINE NUMERICAL MACROS')
    tables={}
    def inline_table(match):
        path=ROOT/(match.group(1)+'.tex');body=path.read_text()
        tables[match.group(1)]={'sha256':sha(path)}
        return '% BEGIN INLINE TABLE '+match.group(1)+'\n'+body+'% END INLINE TABLE '+match.group(1)
    source=re.sub(r'\\input\{(paper/tables/[^}]+)\}',inline_table,source)
    bibliography=(PAPER/'ProvGuard.bbl').read_text()
    source=re.sub(r'\\bibliographystyle\{[^}]+\}\s*\\bibliography\{[^}]+\}',
                  lambda m:'% BEGIN INLINE BIBLIOGRAPHY\n'+bibliography+'\n% END INLINE BIBLIOGRAPHY',source)
    colors={};figures={};blocks=[]
    for stem,name in NAMES.items():
        block,proof=vector_macro(ROOT/f'figures/paper/{stem}.pdf',name,colors)
        blocks.append(block);figures[name]=proof
        source=re.sub(r'\\includegraphics\[width=([^\]]+)\]\{figures/paper/'+stem+r'\.pdf\}',
                      lambda m:'\\'+name+'{'+m.group(1)+'}',source)
    color_defs='\n'.join(r'\definecolor{'+name+'}{rgb}{'+','.join(f(v) for v in rgb)+'}' for rgb,name in colors.items())
    artwork='% BEGIN SELF-CONTAINED VECTOR ARTWORK\n'+color_defs+'\n'+'\n\n'.join(blocks)+'\n% END SELF-CONTAINED VECTOR ARTWORK\n'
    source=source.replace(r'\usepackage{graphicx}',r'\usepackage{graphicx}'+'\n'+r'\usepackage{tikz}'+'\n'+r'\usepackage{helvet}')
    source=source.replace(r'\begin{document}',artwork+'\n'+r'\begin{document}',1)
    source=source.replace('% Empirical numbers and tables are generated from final artifacts.',
        '% Self-contained snapshot: numbers, tables, seven vector figures, and references are embedded.\n% Requires only standard IEEEtran and LaTeX packages; no project input/image/bib files.\n% Rebuild with one normal LaTeX compilation command (rerun for cross-references).')
    assert not re.search(r'\\(?:input|include|includegraphics|bibliography|bibliographystyle)\s*[\[{]',source)
    source_path.write_text(source,encoding='utf-8')
    report={'status':'CONVERTED','modular_source_sha256':sha(modular),'single_tex_sha256':sha(source_path),
            'inline_table_sources':tables,'inline_numerical_macros_sha256':sha(PAPER/'results_macros.tex'),
            'inline_bibliography_sha256':sha(PAPER/'ProvGuard.bbl'),'inline_figures':figures,
            'external_project_file_references':0,'standard_library_dependencies':['IEEEtran','tikz','helvet','cite','amsmath','amssymb','amsfonts','graphicx','textcomp','xcolor','booktabs','url']}
    (PAPER/'single_tex_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'single_tex_bytes':source_path.stat().st_size,'inline_figures':len(figures),'inline_tables':len(tables),'external_project_file_references':0}))

if __name__=='__main__':main()
