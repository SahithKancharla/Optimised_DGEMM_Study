#include <stdio.h>
#include <stdint.h>
#include <omp.h>
#include <stdlib.h>

static const int BASE    = 128;    // same cutoff as your real code
static const int SIZES[] = {64, 128, 256, 512, 1024, 2048, 4096, 8192};
static const int NSIZES  = sizeof(SIZES) / sizeof(*SIZES);

uint64_t counter;
char    *stack_base;       // address of a local in main
size_t   max_stack_usage;  // peak |stack_base - current_frame|

// Dummy “Strassen” that only recurses
void strassen_count(int n) {
    char marker;
    size_t usage = (stack_base > &marker)
            ? (size_t)(stack_base - &marker)
            : (size_t)(&marker - stack_base);
    if (usage > max_stack_usage)
        max_stack_usage = usage;



    counter++;
    if (n <= BASE) return;
    int k = n / 2;
    // #pragma omp task shared(counter,max_stack_usage)
        strassen_count(k);
    // #pragma omp task shared(counter,max_stack_usage)
        strassen_count(k);
    // #pragma omp task shared(counter,max_stack_usage)
        strassen_count(k);
    // #pragma omp task shared(counter,max_stack_usage)
        strassen_count(k);
    // #pragma omp task shared(counter,max_stack_usage)
        strassen_count(k);
    // #pragma omp task shared(counter,max_stack_usage)
        strassen_count(k);
    // #pragma omp task shared(counter,max_stack_usage)
        strassen_count(k);

    // #pragma omp taskwait
}

int main(void) {
    char base_marker;
    stack_base = &base_marker;
    // omp_set_num_threads(32);


    printf("    n    calls      time (s)      avg time/call (µs)        max_stack(bytes)\n");
    printf("-----------------------------------------------------------------------------\n");
    for (int i = 0; i < NSIZES; i++) {
        int n = SIZES[i];
        counter = 0;
        max_stack_usage = 0;

        double t0 = omp_get_wtime();
        strassen_count(n);
        double t1 = omp_get_wtime();

        double tot = t1 - t0;
        double avg_us = (tot / (double)counter) * 1e6;

        printf("%5d %10llu  %12.6f  %16.3f %16zu\n",
               n,
               (unsigned long long)counter,
               tot,
               avg_us,  
               max_stack_usage);
    }
    return 0;
}
