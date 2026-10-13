# Dumas vs. classical Strassen under row-correlated catastrophic ill-conditioning

Resolves the open question in paper `main.tex` Sections 6.4/9/10
(Future Work): does the Dumas et al. (2024) variant's accuracy
advantage narrow the >1400% max-componentwise-error catastrophic case
found elsewhere in this project under row-scaled ill-conditioning?

Added `fill_row_scaled_ill_cond` to `src/avx2/ladder/accuracy_test.c`
(the exact per-row-magnitude-scale construction from
`src/accuracy/distributions.c`'s `dist_wide_dynamic_range`, ported
row-major), run at $N{=}1024$ across 15 seeds, depths 1-3.

## Files

- `row_scaled_seeds_raw.csv` -- raw frob_rel/max_rel per seed/method/depth
- `analyze_row_scaled.py` / `row_scaled_analysis_output.txt` -- win-count summary

## Result

Frobenius-relative error: Dumas lower than classical Strassen in
**15/15 seeds at every depth** -- its norm-wise advantage holds even
under this harsher distribution, extending Section 6.4's finding.

Max-componentwise-relative error (the metric that actually defines the
catastrophic case): **not a reliable win** -- Dumas lower in only
8-9/15 seeds per depth, close to a coin flip, consistent with its
error bound being norm-type, not componentwise (Higham 1990; Ballard
et al. 2016). Dumas's *mean* componentwise error is still noticeably
lower (e.g. depth 3: 26.6 vs. 47.2), but that's driven by lighter
worst-case outliers, not a consistent per-instance advantage -- medians
are nearly identical (depth 3: 5.71 vs. 5.78).
