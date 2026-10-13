# Paper draft

`main.tex` + `references.bib` is a full draft of the paper — the
"spine" (structure, argument, and every headline number) is complete
and locked in, not a scaffold. Structured around FINDINGS.md Part L.3's
three headline contributions (the correctness audit, §4; the
virtualization measurement-bias case study, §5; and the Dumas et al.
2024 Strassen variant from Part M, §6) with everything else as
supporting/appendix material (§7). Every section — abstract,
introduction, related work, methodology, all three headline sections
(including a worked numerical example in §4 and a settled $n=30$
statistical-protocol target in §3.2), supporting material, discussion/
threats, conclusion, and acknowledgments — is written prose with real
numbers pulled from FINDINGS.md and the `revalidation/` CSVs, not
placeholder headers.

**Reworked to ACM TOMS style 2026-10-12** (see
`../../PAPER_TARGET_VENUE.md`, one level above the repo, for the venue
decision/rationale): real `acmart` class (`acmsmall` format, ACM's
Transactions-family format), a plain descriptive title (TOMS's own
author guidance explicitly says to avoid "cute" titles), an
impersonal 150–200-word abstract with no first person and no citation
numbers (TOMS-specific rules, not general ACM style), CCS concepts +
keywords in place of the old plain-`article` preamble, and
`\bibliographystyle{ACM-Reference-Format}`. Only the frontmatter
changed — the section content is the same research. `main.pdf` is a
compiled preview (via `tectonic`, see below) — **real acmart
compilation, not a guess**: zero LaTeX errors, zero unresolved
cross-references or citations in the final pass, only cosmetic
underfull-hbox/font-substitution warnings (verified 2026-10-12) — safe
to delete and regenerate, not something to hand-edit.

## What's still `\todo{}` — genuinely only you can fill these in

Down to 2 markers:
- §9's top-level reproducibility README + license + Zenodo DOI (the
  per-directory READMEs already exist; this is the one remaining
  end-to-end step, per FINDINGS.md Part L.3.2).
- Any additional funding/advisor/reviewer acknowledgments beyond the
  `glogin.cs.vt.edu` access note already there (now inside `\begin{acks}`).

Also worth a human pass before actually submitting: the ACM metadata
placeholders near the top of `main.tex` (`\acmVolume`, `\acmNumber`,
`\acmDOI`, etc. — assigned by ACM at acceptance, not fillable now) and
the CCS concept codes (best-effort, not re-verified against ACM's live
classification tool at dl.acm.org/ccs).

## Using this in Overleaf

1. Go to Overleaf → New Project → Upload Project, and upload this
   `paper/` folder as a zip (`main.tex` + `references.bib`; drop
   `main.pdf`, Overleaf will rebuild it). Or: New Project → Blank
   Project, then upload the two files individually.
2. Overleaf has ACM's official `acmart` template preinstalled, so this
   compiles with zero extra setup there — same as the plain-`article`
   version before it. Default `pdflatex` compiler is fine.
3. To keep iterating locally instead without a system LaTeX install:
   `tectonic main.tex` (see below) — or, with a full TeX Live install,
   `pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex`.

## Venue

Decided: **ACM TOMS**, pursuing the RCR track (see
`../../PAPER_TARGET_VENUE.md` for the full rationale and
pre-submission checklist). CCPE (Wiley) is the explicit fallback if
TOMS doesn't work out.

## Local compile note (no system LaTeX install required)

Verified to compile clean (real `acmart`/`acmsmall`, zero errors) using
a standalone `tectonic` binary, which needs no system TeX Live install
and no root/admin access:
```
curl -L -o tectonic.tar.gz \
  https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.15.0/tectonic-0.15.0-x86_64-unknown-linux-gnu.tar.gz
tar xzf tectonic.tar.gz
./tectonic main.tex
```
