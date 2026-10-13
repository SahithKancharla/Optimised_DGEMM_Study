#include <stdio.h>
#include <stdint.h>
#include <omp.h>

static const int BASE = 128;
static const int SIZES[] = {64, 128, 256, 512, 1024, 2048, 4096, 8192};
static const int NSIZES = sizeof(SIZES)/sizeof(*SIZES);

static uint64_t init_calls;
static uint64_t total_bytes;
char    *stack_base;       // address of a local in main
size_t   max_stack_usage;  // peak |stack_base - current_frame|

// Stub of your allocator
void* initializeMatrix1D(int n) {
    init_calls++;
    total_bytes += (uint64_t)n * (uint64_t)n * sizeof(double);
    return NULL; 
}

// Stubbed recursive structure
void strassen_mem_count(int n) {
    char marker;
    size_t usage = (stack_base > &marker)
            ? (size_t)(stack_base - &marker)
            : (size_t)(&marker - stack_base);
    if (usage > max_stack_usage)
        max_stack_usage = usage;

    if (n <= BASE) {
        initializeMatrix1D(n);
        return;
    }

    int k = n/2;
    for (int i = 0; i < 8; i++)
        initializeMatrix1D(k);
    for (int i = 0; i < 10; i++)
        initializeMatrix1D(k);

    // #pragma omp task shared(max_stack_usage)
        strassen_mem_count(k);
    // #pragma omp task shared(max_stack_usage)
        strassen_mem_count(k);
    // #pragma omp task shared(max_stack_usage)
        strassen_mem_count(k);
    // #pragma omp task shared(max_stack_usage)
        strassen_mem_count(k);
    // #pragma omp task shared(max_stack_usage)
        strassen_mem_count(k);
    // #pragma omp task shared(max_stack_usage)
        strassen_mem_count(k);
    // #pragma omp task shared(max_stack_usage)
        strassen_mem_count(k);

    // #pragma omp taskwait

    for (int i = 0; i < 4; i++)
        initializeMatrix1D(k);

    initializeMatrix1D(n);
}

int main(void) {
    char base_marker;
    stack_base = &base_marker;
    // omp_set_num_threads(32);



    printf(" n    init_calls    total_bytes   (MiB)        max_stack(bytes)\n");
    printf("---------------------------------------------------------------\n");
    for (int i = 0; i < NSIZES; i++) {
        int n = SIZES[i];
        init_calls = 0;
        total_bytes = 0;
        max_stack_usage=0;
        strassen_mem_count(n);
        double mib = (double)total_bytes / (1024.0 * 1024.0);
        printf("%4d %12llu %12llu   %8.2f %16zu\n",
            n,
            (unsigned long long)init_calls,
            (unsigned long long)total_bytes,
            mib, max_stack_usage
        );
    }
    return 0;
}