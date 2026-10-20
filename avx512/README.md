# AVX-512 pipeline

Runs the audited paper's real AVX-512 path (`audited_submission/avx/kernel.c`,
`matmul.c`, `strassen.c`) and an AVX2 baseline on the same machine, checks
correctness, and stores everything in a run folder that merges with runs from
other machines.

```bash
bash avx512/run_avx512.sh                 # quick (sizes 256-2048, best-of-3)
bash avx512/run_avx512.sh --full          # sizes 64-4096, best-of-5
bash avx512/run_avx512.sh --full --reps 14 --tag plugged-in   # closer to paper protocol
# from native Windows (delegates into WSL2):
powershell -ExecutionPolicy Bypass -File .\reproduce.ps1 -Avx512 [-Full]
```

Needs Linux/WSL2, gcc + OpenMP, and a CPU exposing `avx512f` (the script exits
with a clear message otherwise). A failing correctness suite stops the run.

## What it writes
`revalidation/benchmarks/<date>_<host>-<cpu>-avx512/` -- `metadata.txt`,
`avx512.csv` + `avx2.csv` (N,T,algo seconds -- same schema as older runs),
`*_detail.csv` (median, GFLOP/s, relative error), `correctness_avx512.txt`,
`run_info.csv`, `raw_output.txt`, `lscpu.txt`, driver copy.
One summary row is appended to `results/avx512_run_log.csv` (gitignored,
per machine).

## Using other machines
On each AVX-512 system: clone, run the script, then copy its run folder (or
`results/avx512_run_log.csv`) back. Combine everything:

```bash
python3 tools/combine_results.py --extra /path/to/copied/run_folders
```
Output in `results/combined/`: `all_runs_long.csv`, `best_by_machine.csv`,
`run_info_all.csv`, `combined.xlsx` (needs `pip install openpyxl`; the CSVs
open in Excel regardless).

Notes: laptops/VMs throttle and add noise -- plug in, use high-performance
power mode. Tiger Lake and similar client chips have one 512-bit FMA unit per
core, so AVX-512 gains over AVX2 are smaller than on server Xeons.

## Protocol (n=30)
Default is the project standard in `../PROTOCOL.md`: 30 independent launches per
(N,T), AVX-512/AVX2 interleaved, bootstrap CIs and Holm-corrected tests
(`tools/analyze_n30.py` -> `n30_summary.csv`, `n30_comparison.csv`).
`--smoke` runs n=3 for a wiring check only. Override with `--runs`, `--reps`,
`--sizes`, `--threads`. Replace the "Quick/Full" examples above accordingly:
`bash avx512/run_avx512.sh` is now the full n=30 run.
