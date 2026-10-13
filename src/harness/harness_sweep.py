#!/usr/bin/env python3
import subprocess, sys, csv, re

SIZES = [64, 128, 256, 512, 1024]
THREADS_MULTI = [1, 4, 8]

# (algo, is_strassen, needs_threads, max_size_or_None)
ALGOS = [
    ("naive",                          False, False, 512),
    ("naive_unrolled",                 False, False, None),
    ("blocked",                        False, False, None),
    ("blocked_unrolled",               False, False, None),
    ("strassen_pure",                  True,  False, 512),
    ("strassen_hybrid",                True,  False, None),
    ("strassen_hybrid_unrolled",       True,  False, None),
    ("parallel_blocked",               False, True,  None),
    ("parallel_blocked_unrolled",      False, True,  None),
    ("parallel_strassen_hybrid",       True,  True,  None),
    ("parallel_strassen_hybrid_unrolled", True, True, None),
    ("parallel_strassen_pure",         True,  True,  512),
    ("avx2_blocked",                   False, True,  None),
    ("avx2_strassen",                  True,  True,  None),
]

PERF_GROUP_A = "cycles,instructions,branches,branch-misses"
PERF_GROUP_B = "cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses,L1-dcache-stores,dTLB-loads,dTLB-load-misses"

def run(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)

def parse_run_one(stdout):
    # algo,N,T,time_sec,calls,bytes
    parts = stdout.strip().split('\n')[-1].split(',')
    return float(parts[3]), int(parts[4]), int(parts[5])

def parse_perf(stderr):
    out = {}
    for line in stderr.strip().split('\n'):
        parts = line.split(',')
        if len(parts) < 3:
            continue
        val, _, event = parts[0], parts[1], parts[2]
        event = event.replace(':u', '').replace(':k', '')
        if val == '<not supported>' or val == '<not counted>':
            out[event] = None
        else:
            try:
                out[event] = int(val)
            except ValueError:
                pass
    return out

def main():
    rows = []
    fieldnames = ["algo", "N", "T", "time_sec", "calls", "bytes_allocated",
                  "cycles", "instructions", "branches", "branch_misses",
                  "cache_references", "cache_misses",
                  "L1_dcache_loads", "L1_dcache_load_misses", "L1_dcache_stores",
                  "dTLB_loads", "dTLB_load_misses"]

    total = 0
    for algo, is_strassen, needs_threads, max_size in ALGOS:
        for N in SIZES:
            if max_size and N > max_size:
                continue
            threads = THREADS_MULTI if needs_threads else [1]
            for T in threads:
                total += 1
    print(f"# {total} configs to run", file=sys.stderr)

    done = 0
    for algo, is_strassen, needs_threads, max_size in ALGOS:
        for N in SIZES:
            if max_size and N > max_size:
                continue
            threads = THREADS_MULTI if needs_threads else [1]
            for T in threads:
                done += 1
                print(f"# [{done}/{total}] {algo} N={N} T={T}", file=sys.stderr)

                r = run(f"./run_one_plain {algo} {N} {T}")
                time_sec, _, _ = parse_run_one(r.stdout)

                counters = {}
                for grp in (PERF_GROUP_A, PERF_GROUP_B):
                    rp = run(f"perf stat -e {grp} -x, -- ./run_one_plain {algo} {N} {T}")
                    counters.update(parse_perf(rp.stderr))

                calls = bytes_alloc = 0
                if is_strassen:
                    rc = run(f"./run_one_counted {algo} {N} {T}")
                    _, calls, bytes_alloc = parse_run_one(rc.stdout)

                rows.append({
                    "algo": algo, "N": N, "T": T, "time_sec": time_sec,
                    "calls": calls, "bytes_allocated": bytes_alloc,
                    "cycles": counters.get("cycles"),
                    "instructions": counters.get("instructions"),
                    "branches": counters.get("branches"),
                    "branch_misses": counters.get("branch-misses"),
                    "cache_references": counters.get("cache-references"),
                    "cache_misses": counters.get("cache-misses"),
                    "L1_dcache_loads": counters.get("L1-dcache-loads"),
                    "L1_dcache_load_misses": counters.get("L1-dcache-load-misses"),
                    "L1_dcache_stores": counters.get("L1-dcache-stores"),
                    "dTLB_loads": counters.get("dTLB-loads"),
                    "dTLB_load_misses": counters.get("dTLB-load-misses"),
                })

    with open("harness_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"# wrote harness_results.csv ({len(rows)} rows)", file=sys.stderr)

if __name__ == "__main__":
    main()
