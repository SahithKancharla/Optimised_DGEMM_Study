# Paper draft

`main.tex` + `references.bib` is a full draft of the paper, structured
around FINDINGS.md Part L.3's three headline contributions (the
correctness audit, §4; the virtualization measurement-bias case study,
§5; and the Dumas et al. 2024 Strassen variant from Part M, §6) with
everything else as supporting/appendix material (§7). Every section is
drafted from real, already-executed results — abstract, introduction,
related work, methodology, all three headline sections, supporting
material, discussion/threats, and conclusion are written prose with real
numbers pulled from FINDINGS.md and the `revalidation/` CSVs, not
placeholder headers. `main.pdf` is a compiled preview (built with
[tectonic](https://tectonic-typesetting.github.io/), last verified clean
— zero errors, one sub-point overfull-hbox warning — 2026-09-09) — safe
to delete and regenerate, not something to hand-edit.

## What's still `\todo{}` — genuinely only you can fill these in

- Author name(s), affiliation(s), corresponding email (top of `main.tex`).
- §3.2's exact repetition count `n` for the bootstrap-CI protocol (a
  placeholder `n=20`–`30` is suggested, following Davis 2025).
- §4's optional worked numerical example for the $C=BA$-not-$AB$ bug
  (nice-to-have for concreteness, not required).
- §9's top-level reproducibility README + license + Zenodo DOI (the
  per-directory READMEs already exist; this is the one remaining
  end-to-end step, per FINDINGS.md Part L.3.2).
- Acknowledgments (end of the document).

Everything else in the document is complete prose, not a stub.

## Using this in Overleaf

1. Go to Overleaf → New Project → Upload Project, and upload this
   `paper/` folder as a zip (`main.tex` + `references.bib`; drop
   `main.pdf`, Overleaf will rebuild it). Or: New Project → Blank
   Project, then upload the two files individually.
2. Overleaf auto-detects `main.tex` as the root document and compiles
   with `pdflatex` by default — that matches what this file was written
   and tested against (via `tectonic`, a standards-compliant LaTeX
   engine; both target the same `article`-class output). No extra
   Overleaf settings should be needed.
3. To keep iterating locally instead: `pdflatex main.tex && bibtex main
   && pdflatex main.tex && pdflatex main.tex` (two passes after bibtex
   so citations and the table of contents settle), or simply
   `tectonic main.tex` if you have the standalone engine (no full
   TeX Live install needed — see the one-off install note below).

## What's real vs. placeholder right now

Every `\todo{...}` marker is a real gap — either a section that needs
prose written, or a specific table/figure that needs pulling from a
named CSV in `revalidation/`. Everything **not** marked `\todo` is
already drafted from real, executed results (the correctness-bug list
in §4, the virtualization case study's claim/method/result in §5, and
all of §6 on the Dumas 2024 variant, including the performance table)
— re-derive exact numbers from the source CSVs named inline rather than
retyping them from the PDF if anything changes upstream.

## Venue

Not yet chosen. FINDINGS.md Part L.3 recommends, in order: ACM TOMS's
Replicated Computational Results track (best fit for §4 alone), the
PMBS workshop (SC-colocated; best fit for §5), or an arXiv preprint
immediately regardless of either. Swapping `\documentclass{article}`
for `acmart` (with the appropriate journal/proceedings options) is the
only preamble change either ACM venue would require — the section
structure and content below the preamble should not need to change for
that swap.

## Local compile note (no system LaTeX install required)

This was verified to compile clean (zero warnings, `article` class +
`amsmath`/`booktabs`/`graphicx`/`hyperref`/`xcolor`/`microtype`) using a
standalone `tectonic` binary, which needs no system TeX Live install and
no root/admin access — useful if iterating outside Overleaf:
```
curl -L -o tectonic.tar.gz \
  https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.15.0/tectonic-0.15.0-x86_64-unknown-linux-gnu.tar.gz
tar xzf tectonic.tar.gz
./tectonic main.tex
```
