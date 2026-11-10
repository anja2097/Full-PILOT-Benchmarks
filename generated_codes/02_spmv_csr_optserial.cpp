/* ANALYSIS: SpMV CSR kernel, memory-bandwidth bound due to irregular x accesses.
   Transformations: 64‑byte aligned allocations, __restrict__ pointers, software prefetch,
   blocked rows (BLOCK_ROWS) for L2 fit, 4‑way accumulator unrolling to reduce dependence. */

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>

#ifndef BLOCK_ROWS
#define BLOCK_ROWS 256          // fits L2 (≈512 KB) with 16 nnz/row → 4 KB per block
#endif
#ifndef PREFETCH_DIST
#define PREFETCH_DIST 16        // distance for software prefetch of x[]
#endif

static inline void spmv_kernel(const long N,
                               const long * __restrict__ row_ptr,
                               const int * __restrict__ col,
                               const double * __restrict__ val,
                               const double * __restrict__ x,
                               double * __restrict__ y,
                               const double * __restrict__ b) {
    for (long row_start = 0; row_start < N; row_start += BLOCK_ROWS) {
        long row_end = row_start + BLOCK_ROWS;
        if (row_end > N) row_end = N;
        for (long i = row_start; i < row_end; ++i) {
            double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
            long k = row_ptr[i];
            long k_end = row_ptr[i + 1];
            for (; k + 3 < k_end; k += 4) {
                __builtin_prefetch(&x[col[k + PREFETCH_DIST]], 0, 1);
                __builtin_prefetch(&x[col[k + PREFETCH_DIST + 1]], 0, 1);
                __builtin_prefetch(&x[col[k + PREFETCH_DIST + 2]], 0, 1);
                __builtin_prefetch(&x[col[k + PREFETCH_DIST + 3]], 0, 1);

                sum0 += val[k]     * x[col[k]];
                sum1 += val[k + 1] * x[col[k + 1]];
                sum2 += val[k + 2] * x[col[k + 2]];
                sum3 += val[k + 3] * x[col[k + 3]];
            }
            for (; k < k_end; ++k) {
                sum0 += val[k] * x[col[k]];
            }
            double sum = sum0 + sum1 + sum2 + sum3;
            y[i] = sum + b[i];
        }
    }
}

int main() {
    const int N = 1 << 21;          // 2 097 152 rows
    const int NNZ_PER_ROW = 16;
    const int NUM_ITERS = 30;
    const long NNZ = static_cast<long>(N) * NNZ_PER_ROW;

    // 64‑byte aligned allocations
    long *row_ptr = (long*)aligned_alloc(64, (N + 1) * sizeof(long));
    int *col = (int*)aligned_alloc(64, NNZ * sizeof(int));
    double *val = (double*)aligned_alloc(64, NNZ * sizeof(double));
    double *x = (double*)aligned_alloc(64, N * sizeof(double));
    double *y = (double*)aligned_alloc(64, N * sizeof(double));
    double *b = (double*)aligned_alloc(64, N * sizeof(double));

    if (!row_ptr || !col || !val || !x || !y || !b) {
        std::fprintf(stderr, "Allocation failed\n");
        std::exit(1);
    }

    std::srand(42);
    for (int i = 0; i < N; ++i) {
        row_ptr[i] = static_cast<long>(i) * NNZ_PER_ROW;
        double row_sum = 0.0;
        for (int k = 0; k < NNZ_PER_ROW; ++k) {
            long idx = row_ptr[i] + k;
            col[idx] = std::rand() % N;
            val[idx] = 0.1 + static_cast<double>(std::rand()) / RAND_MAX;
            row_sum += val[idx];
        }
        double scale = 0.9 / row_sum;
        for (int k = 0; k < NNZ_PER_ROW; ++k) {
            val[row_ptr[i] + k] *= scale;
        }
    }
    row_ptr[N] = NNZ;

    for (int i = 0; i < N; ++i) x[i] = static_cast<double>(std::rand()) / RAND_MAX;
    for (int i = 0; i < N; ++i) b[i] = static_cast<double>(std::rand()) / RAND_MAX;

    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < NUM_ITERS; ++iter) {
        spmv_kernel(N, row_ptr, col, val, x, y, b);
        std::swap(x, y);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    std::printf("Tiempo de ejecucion de la SpMV (%d iters): %.2f ms\n", NUM_ITERS, elapsed.count());

    double checksum = 0.0;
    for (int i = 0; i < N; ++i) {
        checksum += x[i] * static_cast<double>(i % 11 + 1);
    }
    std::printf("Checksum: %.6f\n", checksum);

    std::free(row_ptr);
    std::free(col);
    std::free(val);
    std::free(x);
    std::free(y);
    std::free(b);
    return 0;
}