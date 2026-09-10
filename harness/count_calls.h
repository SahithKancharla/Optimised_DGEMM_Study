#ifndef COUNT_CALLS_H
#define COUNT_CALLS_H

/*
 * Opt-in recursive-call and allocation-byte counters for the Strassen
 * variants. Compiled out entirely unless COUNT_CALLS is defined at build
 * time (-DCOUNT_CALLS), so normal production builds of Serial/Parallel/avx
 * are byte-for-byte unaffected. Each instrumented .c file declares its own
 * uniquely-named counter pair via COUNT_CALLS_DEFINE(name) and increments
 * them with COUNT_CALL() / COUNT_BYTES(n) -- kept as simple globals (not
 * atomics) for the Serial files, and OpenMP-atomic for the Parallel/task-
 * based files, matched to how each file is actually structured.
 *
 * Recursion depth is NOT tracked at runtime: for Strassen, depth is fully
 * determined by (starting N, cutoff/base case) via
 * depth = ceil(log2(N / base_n)), so the harness computes it analytically
 * instead of adding tracking overhead.
 */

#define COUNT_CALLS_DEFINE(name) \
    unsigned long long g_calls_##name = 0; \
    unsigned long long g_bytes_##name = 0;

#define COUNT_CALLS_DEFINE_ATOMIC(name) \
    unsigned long long g_calls_##name = 0; \
    unsigned long long g_bytes_##name = 0;

#ifdef COUNT_CALLS
  #define COUNT_CALL(name)          do { g_calls_##name++; } while (0)
  #define COUNT_BYTES(name, nbytes) do { g_bytes_##name += (unsigned long long)(nbytes); } while (0)
  #define COUNT_CALL_ATOMIC(name)          _Pragma("omp atomic") g_calls_##name++
  #define COUNT_BYTES_ATOMIC(name, nbytes) do { unsigned long long _b = (unsigned long long)(nbytes); _Pragma("omp atomic") g_bytes_##name += _b; } while (0)
#else
  #define COUNT_CALL(name)                 do {} while (0)
  #define COUNT_BYTES(name, nbytes)        do {} while (0)
  #define COUNT_CALL_ATOMIC(name)          do {} while (0)
  #define COUNT_BYTES_ATOMIC(name, nbytes) do {} while (0)
#endif

#endif
