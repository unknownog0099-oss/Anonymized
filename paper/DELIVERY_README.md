# ProvGuard single-file paper

ProvGuard.tex contains the complete anonymous six-page IEEE conference paper:
153 numerical macro definitions, three manuscript tables, fourteen references,
and seven native TikZ vector figures. Four figures are used in the manuscript;
the additional three figures remain embedded as optional drawing macros.
No external project files are needed to compile this source.

## Compile

Use a standard TeX distribution with IEEEtran, PGF/TikZ, Helvetica, and the other
packages declared in the preamble. From a folder containing only this TeX file:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error ProvGuard.tex
pdflatex -interaction=nonstopmode -halt-on-error ProvGuard.tex
```

Alternatively, use `tectonic ProvGuard.tex`. No BibTeX command is required.
The source was tested in an isolated folder containing only ProvGuard.tex:
it compiled to six pages with identical extracted text, fourteen resolved
references, zero overfull boxes, and zero missing glyphs. Every final page
was visually inspected. paper/single_tex_audit.json records the proof.

## Original project workflow

The project retains data-linked editable sources in figures/paper/,
paper/tables/, paper/results_macros.tex, and ProvGuard.bib.
They are generation and audit inputs, rather than compilation dependencies.
The previous modular manuscript and PDF are preserved as
paper/ProvGuard_modular.tex and paper/ProvGuard_modular.pdf.

To recompile the existing final source in the workspace:

```powershell
.\.venv\Scripts\python.exe scripts\build_paper.py
.\.venv\Scripts\python.exe scripts\audit_paper.py
.\.venv\Scripts\python.exe scripts\verify_single_tex.py
```

To regenerate the complete snapshot, first edit the preserved modular source,
then run scripts/build_paper_assets.py, scripts/make_single_tex.py, and the
three commands above. This regeneration overwrites the root single-file source.
The archived deep experimental audit is paper/numerical_audit_modular.json;
the current numerical audit checks embedded content against unchanged evidence.
No model training or experimental analysis is required for ordinary compilation.
