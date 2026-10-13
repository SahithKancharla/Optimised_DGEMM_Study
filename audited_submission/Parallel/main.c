#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#define RUNS 14
#define DISCARD 2

extern double* mm_parallel_blocked(int nthreads, int rows, int cols_a, int cols_b);
extern double* mm_parallel_blocked_unrolled(int nthreads, int rows, int cols_a, int cols_b);
extern double* mm_parallel_strassen(int nthreads, int rows, int cols_a, int cols_b);
extern double* mm_parallel_strassen_unrolled(int nthreads, int rows, int cols_a, int cols_b);
extern double* mm_parallel_strassen_naive(int nthreads, int rows, int cols_a, int cols_b);

int compare(const void *a, const void *b) {
    double da = *(const double *)a;
    double db = *(const double *)b;
    return (da > db) - (da < db);
}

void prepare_output_dirs() {
    struct stat st = {0};

    // Remove and recreate "Values"
    if (stat("Values", &st) == 0) {
        system("rm -rf Values");
    }
    if (mkdir("Values", 0777) != 0) {
        perror("Failed to create Values directory");
        exit(1);
    }

    // Remove and recreate "Results"
    if (stat("Results", &st) == 0) {
        system("rm -rf Results");
    }
    if (mkdir("Results", 0777) != 0) {
        perror("Failed to create Results directory");
        exit(1);
    }

    // Remove and recreate "Data"
    if (stat("Data", &st) == 0) {
        system("rm -rf Data");
    }
    if (mkdir("Data", 0777) != 0) {
        perror("Failed to create Data directory");
        exit(1);
    }
}


void initialize_and_write_matrices(int rows, int cols_a, int cols_b) {
    double *a = (double *)malloc(rows * cols_a * sizeof(double));
    double *b = (double *)malloc(cols_a * cols_b * sizeof(double));

    if (!a || !b) {
        fprintf(stderr, "Memory allocation failed for matrices.\n");
        exit(1);
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols_a; j++) {
            a[i * cols_a + j] = i + j;
        }
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < cols_a; i++) {
        for (int j = 0; j < cols_b; j++) {
            b[i * cols_b + j] = i * j;
        }
    }

    FILE *fa = fopen("Data/matrix_a.bin", "wb");
    FILE *fb = fopen("Data/matrix_b.bin", "wb");

    if (!fa || !fb) {
        fprintf(stderr, "Failed to open file for writing matrices.\n");
        exit(1);
    }

    fwrite(a, sizeof(double), rows * cols_a, fa);
    fwrite(b, sizeof(double), cols_a * cols_b, fb);

    fclose(fa);
    fclose(fb);

    free(a);
    free(b);
}

void write_result_matrix(const char *filename, double *c, int rows, int cols) {
    FILE *fr = fopen(filename, "w");
    if (!fr) {
        perror("Error opening result file for writing");
        return;
    }

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            fprintf(fr, "%f ", c[i * cols + j]);
        }
        fprintf(fr, "\n");
    }

    fclose(fr);
}

int read_all_times(const char *filename, double *times, int expected_runs) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Error opening time file");
        return -1;
    }
    for (int i = 0; i < expected_runs; i++) {
        if (fscanf(file, "%lf", &times[i]) != 1) {
            fprintf(stderr, "Failed to read timing value %d from %s\n", i + 1, filename);
            fclose(file);
            return -1;
        }
    }
    fclose(file);
    return 0;
}

void run_and_time(const char *label, const char *result_file,
                  double* (*matmul_func)(int, int, int, int),
                  int nthreads, int rows, int cols_a, int cols_b) {
    double *c = NULL;

    // Run the algorithm multiple times (each call is expected to log time to parallel.txt)
    for (int i = 0; i < RUNS; i++) {
        c = matmul_func(nthreads, rows, cols_a, cols_b);
    }

    double times[RUNS];
    if (read_all_times("Values/parallel.txt", times, RUNS) == 0) {
        qsort(times, RUNS, sizeof(double), compare);
        double sum = 0.0;
        for (int i = DISCARD; i < RUNS - DISCARD; i++) {
            sum += times[i];
        }
        double avg = sum / (RUNS - 2 * DISCARD);
        printf("%s Average Time: %f seconds\n\n", label, avg);
    }
    write_result_matrix(result_file, c, rows, cols_b);

    free(c);
}

int main(int argc, char *argv[]) {
    if (argc < 5) {
        printf("Usage: %s <nthreads> <rows> <cols_a> <cols_b> [blocked|blocked_unrolled|strassen|strassen_unrolled|naive|all]\n", argv[0]);
        return 1;
    }

    int nthreads = atoi(argv[1]);
    int rows = atoi(argv[2]);
    int cols_a = atoi(argv[3]);
    int cols_b = atoi(argv[4]);
    char *algorithm = (argc > 5) ? argv[5] : "all";

    omp_set_num_threads(nthreads);
    prepare_output_dirs();
    initialize_and_write_matrices(rows, cols_a, cols_b);

    if (strcmp(algorithm, "blocked") == 0 || strcmp(algorithm, "all") == 0) {
        run_and_time("Parallel Blocked", "Results/result_parallel_blocked.txt",
                     mm_parallel_blocked, nthreads, rows, cols_a, cols_b);
    }

    if (strcmp(algorithm, "blocked_unrolled") == 0 || strcmp(algorithm, "all") == 0) {
        run_and_time("Parallel Blocked Unrolled", "Results/result_parallel_blocked_unrolled.txt",
                     mm_parallel_blocked_unrolled, nthreads, rows, cols_a, cols_b);
    }

    if (strcmp(algorithm, "strassen_base") == 0 || strcmp(algorithm, "all") == 0) {
        run_and_time("Parallel Strassen Base", "Results/result_parallel_strassen.txt",
                     mm_parallel_strassen, nthreads, rows, cols_a, cols_b);
    }

    if (strcmp(algorithm, "strassen_base_unrolled") == 0 || strcmp(algorithm, "all") == 0) {
        run_and_time("Parallel Strassen Base Unrolled", "Results/result_parallel_strassen_unrolled.txt",
                     mm_parallel_strassen_unrolled, nthreads, rows, cols_a, cols_b);
    }

    if (strcmp(algorithm, "strassen") == 0 || strcmp(algorithm, "all") == 0) {
        run_and_time("Parallel Strassen Naive", "Results/result_parallel_naive.txt",
                     mm_parallel_strassen_naive, nthreads, rows, cols_a, cols_b);
    }

    return 0;
}
