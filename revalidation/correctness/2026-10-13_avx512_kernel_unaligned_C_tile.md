# AVX-512 kernel: aligned C-tile access crashes when N % 8 != 0

Found 2026-10-13 on the first-ever execution of `audited_submission/avx/kernel.c`
on real AVX-512 hardware (11th Gen Intel i5-1145G7, WSL2 Ubuntu 26.04, gcc 15.2,
`-O3 -march=skylake-avx512`).

**Symptom:** `test_local.c` linked against `kernel.c` segfaults (SIGSEGV, general
protection fault) at N=25. AddressSanitizer: `_mm512_load_pd` at kernel.c:66
(`LD_FULL`, full 24-row fast path).

**Cause:** `kernel_24x6` loads/stores its C tile at `C + col*ldC (+0/8/16)` with
`_mm512_load_pd` / `_mm512_store_pd` (require 64-byte alignment). `ldC = N`, so
alignment holds only when N is a multiple of 8. The paper's benchmark sizes
(powers of two) never hit this; any other N with a full 24-row tile faults.
The earlier "allocate C with 64-byte alignment" note in `test_local.c` is
necessary but not sufficient.

**Fix (kernel.c):** C-tile loads (LD_FULL) -> `_mm512_loadu_pd`, stores (ST_FULL)
-> `_mm512_storeu_pd`. Packed-A loads stay aligned (static 64B-aligned buffer,
offsets are multiples of 192 B). No measurable cost on aligned data.

**Verification:** `test_local.c` + `kernel.c` + `strassen.c`: ALL PASS, 17 matmul
sizes (1..517) + 17 Strassen sizes, maxdiff 0.
