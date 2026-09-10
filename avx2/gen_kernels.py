#!/usr/bin/env python3
"""Generates one self-contained AVX2 microkernel sweep source file per (MR,NR)
shape from kernel_template.c.in. MR must be a multiple of 4 (AVX2 YMM = 4
doubles); NR is unconstrained. Run with no args to (re)generate all shapes
listed in SHAPES."""
import os

SHAPES = [
    (4, 8), (4, 12),
    (8, 4), (8, 6), (8, 8),
    (12, 4), (12, 6),
    (16, 4), (16, 6),
]

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "kernel_template.c.in")
OUTDIR = os.path.join(HERE, "sweep")

def main():
    with open(TEMPLATE, "r") as f:
        template = f.read()
    os.makedirs(OUTDIR, exist_ok=True)
    manifest = []
    for mr, nr in SHAPES:
        mrv = mr // 4
        acc = mrv * nr
        src = (template
               .replace("{{MR}}", str(mr))
               .replace("{{NR}}", str(nr))
               .replace("{{MRV}}", str(mrv))
               .replace("{{ACC}}", str(acc)))
        fname = f"kernel_{mr}x{nr}.c"
        with open(os.path.join(OUTDIR, fname), "w") as f:
            f.write(src)
        manifest.append((mr, nr, mrv, acc, fname))
        print(f"generated {fname}  (accumulators={acc}, total_regs_est={acc+mrv+1})")
    with open(os.path.join(OUTDIR, "manifest.csv"), "w") as f:
        f.write("mr,nr,mrv,accumulators,total_regs_est,file\n")
        for mr, nr, mrv, acc, fname in manifest:
            f.write(f"{mr},{nr},{mrv},{acc},{acc+mrv+1},{fname}\n")

if __name__ == "__main__":
    main()
