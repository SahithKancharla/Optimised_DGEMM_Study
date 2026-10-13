/* Multi-threaded AVX2 FMA throughput benchmark, native Windows, with
 * EXPLICIT per-thread core pinning via SetThreadAffinityMask -- WSL2/
 * Hyper-V cannot reliably pin to a specific physical core (see
 * classify_cores.c / FINDINGS.md Part A), so this must run natively.
 * Used for two things at once: (1) the P-core/E-core/SMT scaling
 * substitute-NUMA study (no real NUMA on this single-socket machine), and
 * (2) a real MEASURED peak-GFLOP/s figure for %-of-peak analysis, instead
 * of an estimated one.
 *
 * Usage: peak_scaling_native.exe <cpu0> <cpu1> ... <cpuN>
 * Spawns one thread per listed logical CPU index, each pinned via
 * SetThreadAffinityMask, each running the same register-only FMA loop as
 * topology/classify_cores.c (8 independent accumulator chains, no memory
 * traffic), reports aggregate GFLOP/s across all threads. */
#include <windows.h>
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>

#define ITERS 800000000LL

typedef struct { int cpu; double gflops; } thread_arg_t;

static DWORD WINAPI worker(LPVOID p) {
    thread_arg_t *arg = (thread_arg_t *)p;
    if (arg->cpu < 0 || arg->cpu >= 64) {
        // (DWORD_PTR)1 << cpu is undefined behavior (and wraps mod 64 on
        // real x86 hardware) once cpu >= the bit-width of DWORD_PTR -- that
        // would silently pin to the wrong core instead of failing.
        fprintf(stderr, "cpu index %d out of range (0-63)\n", arg->cpu);
        arg->gflops = -1.0;
        return 1;
    }
    DWORD_PTR mask = (DWORD_PTR)1 << arg->cpu;
    if (!SetThreadAffinityMask(GetCurrentThread(), mask)) {
        fprintf(stderr, "SetThreadAffinityMask failed for cpu %d\n", arg->cpu);
        arg->gflops = -1.0;
        return 1;
    }

    __m256d b = _mm256_set1_pd(0.9999999);
    __m256d a[8], c[8];
    for (int i = 0; i < 8; i++) { a[i] = _mm256_set1_pd(1.0000001 + 0.0000001 * i); c[i] = _mm256_setzero_pd(); }

    LARGE_INTEGER freq, t0, t1;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t0);
    for (long long i = 0; i < ITERS; i++) {
        c[0] = _mm256_fmadd_pd(a[0], b, c[0]); c[1] = _mm256_fmadd_pd(a[1], b, c[1]);
        c[2] = _mm256_fmadd_pd(a[2], b, c[2]); c[3] = _mm256_fmadd_pd(a[3], b, c[3]);
        c[4] = _mm256_fmadd_pd(a[4], b, c[4]); c[5] = _mm256_fmadd_pd(a[5], b, c[5]);
        c[6] = _mm256_fmadd_pd(a[6], b, c[6]); c[7] = _mm256_fmadd_pd(a[7], b, c[7]);
    }
    QueryPerformanceCounter(&t1);
    double secs = (double)(t1.QuadPart - t0.QuadPart) / (double)freq.QuadPart;
    arg->gflops = (double)ITERS * 8.0 * 4.0 * 2.0 / secs / 1e9;

    double sink = 0;
    for (int i = 0; i < 8; i++) { double t[4]; _mm256_storeu_pd(t, c[i]); sink += t[0] + t[1] + t[2] + t[3]; }
    if (sink == 123456.789) fprintf(stderr, "unreachable %f\n", sink);
    return 0;
}

int main(int argc, char **argv) {
    int n = argc - 1;
    if (n < 1) { fprintf(stderr, "usage: %s cpu0 [cpu1 ...]\n", argv[0]); return 1; }
    if (n > MAXIMUM_WAIT_OBJECTS) {
        // WaitForMultipleObjects hard-caps nCount at MAXIMUM_WAIT_OBJECTS
        // (64) and returns WAIT_FAILED immediately (without waiting for
        // anything) above that -- silently producing torn/zero results if
        // unchecked, instead of failing loudly here.
        fprintf(stderr, "too many cpus requested (%d); at most %d supported\n", n, MAXIMUM_WAIT_OBJECTS);
        return 1;
    }

    thread_arg_t *args = malloc(sizeof(thread_arg_t) * n);
    HANDLE *th = malloc(sizeof(HANDLE) * n);
    for (int i = 0; i < n; i++) {
        args[i].cpu = atoi(argv[i + 1]);
        args[i].gflops = 0;
        th[i] = CreateThread(NULL, 0, worker, &args[i], 0, NULL);
        if (!th[i]) {
            // An invalid/NULL handle anywhere in the array poisons the
            // whole WaitForMultipleObjects call below (not just this
            // slot), silently corrupting every other thread's result too.
            fprintf(stderr, "CreateThread failed for cpu %d (GetLastError=%lu)\n",
                    args[i].cpu, (unsigned long)GetLastError());
            return 1;
        }
    }
    if (WaitForMultipleObjects(n, th, TRUE, INFINITE) == WAIT_FAILED) {
        fprintf(stderr, "WaitForMultipleObjects failed (GetLastError=%lu)\n",
                (unsigned long)GetLastError());
        return 1;
    }

    double total = 0;
    int any_failed = 0;
    for (int i = 0; i < n; i++) {
        printf("cpu%d: %.3f GFLOP/s\n", args[i].cpu, args[i].gflops);
        if (args[i].gflops < 0) any_failed = 1; // worker()'s failure sentinel
        else total += args[i].gflops;
    }
    printf("TOTAL (%d threads): %.3f GFLOP/s\n", n, total);
    if (any_failed) {
        fprintf(stderr, "one or more worker threads failed (see cpu errors above); "
                        "TOTAL excludes them and is not a full-thread-count figure\n");
        return 1;
    }
    return 0;
}
