#!/usr/bin/env python3
"""combine_results.py -- merge benchmark runs from many machines into one table.

Reads every run folder under revalidation/benchmarks/ (serial.csv, parallel.csv,
avx2.csv, avx512.csv + metadata.txt) plus every run_info / run_log CSV, and
writes to results/combined/:

  all_runs_long.csv   one row per (machine, kernel, algorithm, N, T): seconds, GFLOP/s
  best_by_machine.csv best GFLOP/s per (machine, kernel, algorithm) and where it occurred
  run_info_all.csv    every run_info.csv / run_log.csv row, columns unioned
  combined.xlsx       the three above as sheets (needs `pip install openpyxl`)

Results from another machine: copy its run folder (the whole
<date>_<host>-<cpu>-avx512/ directory) or its results/*.csv anywhere and pass
the path with --extra; no other setup.

Usage: python3 tools/combine_results.py [--extra DIR ...] [--out DIR]
"""
import argparse, csv, glob, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_CSVS = {"serial.csv": "scalar", "parallel.csv": "scalar-omp",
            "avx2.csv": "avx2", "avx512.csv": "avx512"}


def parse_meta(path):
    meta = {}
    if os.path.exists(path):
        for line in open(path, encoding="utf-8", errors="replace"):
            m = re.match(r"^([a-z0-9_]+):\s*(.*)$", line)
            if m and m.group(1) not in meta:
                meta[m.group(1)] = m.group(2).strip()
    return meta


def find_run_dirs(base):
    if any(os.path.exists(os.path.join(base, f)) for f in RUN_CSVS):
        return [base]
    return sorted(d for d in glob.glob(os.path.join(base, "*")) if os.path.isdir(d)
                  and any(os.path.exists(os.path.join(d, f)) for f in RUN_CSVS))


def num(x):
    try:
        v = float(x)
        return v if v > 0 else None
    except (TypeError, ValueError):
        return None


def long_rows(run_dir):
    name = os.path.basename(os.path.normpath(run_dir))
    date = name[:10] if re.match(r"\d{4}-\d{2}-\d{2}_", name) else ""
    machine = re.sub(r"^\d{4}-\d{2}-\d{2}_", "", name)
    meta = parse_meta(os.path.join(run_dir, "metadata.txt"))
    cpu = meta.get("cpu", "")[:80]
    for fname, isa in RUN_CSVS.items():
        p = os.path.join(run_dir, fname)
        if not os.path.exists(p):
            continue
        for r in csv.DictReader(open(p, encoding="utf-8")):
            N = int(r["N"]); T = int(r.get("T") or 1)
            for algo, v in r.items():
                if algo in ("N", "T"):
                    continue
                s = num(v)
                if s is None:
                    continue  # -1 = not run at this size
                yield {"machine": machine, "run_dir": name, "date": date, "cpu": cpu,
                       "kernel": isa, "algorithm": algo, "N": N, "T": T,
                       "seconds": s, "gflops": round(2.0 * N ** 3 / s / 1e9, 3)}


def write_csv(path, rows, cols=None):
    cols = cols or (list(rows[0].keys()) if rows else [])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra", nargs="*", default=[], help="extra run folders / parents / dirs with run_*.csv")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "combined"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    bases = [os.path.join(ROOT, "revalidation", "benchmarks")] + a.extra
    run_dirs = []
    for b in bases:
        run_dirs += find_run_dirs(b)
    longr = [r for d in run_dirs for r in long_rows(d)]
    longr.sort(key=lambda r: (r["machine"], r["kernel"], r["algorithm"], r["N"], r["T"]))

    best = {}
    for r in longr:
        k = (r["machine"], r["kernel"], r["algorithm"])
        if k not in best or r["gflops"] > best[k]["best_gflops"]:
            best[k] = {"machine": k[0], "kernel": k[1], "algorithm": k[2], "cpu": r["cpu"],
                       "best_gflops": r["gflops"], "at_N": r["N"], "at_T": r["T"], "run_dir": r["run_dir"]}
    best_rows = sorted(best.values(), key=lambda r: (r["machine"], r["kernel"], -r["best_gflops"]))

    info, seen = [], set()
    csvs = []
    for b in bases + [os.path.join(ROOT, "results")]:
        csvs += glob.glob(os.path.join(b, "**", "run_info.csv"), recursive=True)
        csvs += glob.glob(os.path.join(b, "**", "run_log*.csv"), recursive=True)
        csvs += glob.glob(os.path.join(b, "**", "avx512_run_log.csv"), recursive=True)
    for p in sorted(set(csvs)):
        if os.sep + "combined" + os.sep in p:
            continue
        for r in csv.DictReader(open(p, encoding="utf-8")):
            key = tuple(sorted(r.items()))
            if key not in seen:
                seen.add(key); info.append(r)
    info_cols = []
    for r in info:
        for c in r:
            if c not in info_cols:
                info_cols.append(c)

    write_csv(os.path.join(a.out, "all_runs_long.csv"), longr,
              ["machine", "run_dir", "date", "cpu", "kernel", "algorithm", "N", "T", "seconds", "gflops"])
    write_csv(os.path.join(a.out, "best_by_machine.csv"), best_rows)
    write_csv(os.path.join(a.out, "run_info_all.csv"), info, info_cols)

    print(f"{len(run_dirs)} run folders, {len(longr)} data points, {len(info)} run_info rows -> {a.out}")
    machines = sorted({r['machine'] for r in longr})
    print("machines:", ", ".join(machines) or "(none)")

    try:
        import openpyxl
        wb = openpyxl.Workbook(); wb.remove(wb.active)
        for title, rows, cols in (("best_by_machine", best_rows, None),
                                  ("all_runs_long", longr, None),
                                  ("run_info_all", info, info_cols)):
            ws = wb.create_sheet(title)
            cols = cols or (list(rows[0].keys()) if rows else [])
            ws.append(cols)
            for r in rows:
                ws.append([r.get(c, "") for c in cols])
            ws.freeze_panes = "A2"
        wb.save(os.path.join(a.out, "combined.xlsx"))
        print("wrote combined.xlsx")
    except ImportError:
        print("(openpyxl not installed -> skipped .xlsx; CSVs open in Excel directly. "
              "`pip install openpyxl` to also get combined.xlsx)")


if __name__ == "__main__":
    sys.exit(main())
