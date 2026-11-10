/* ANALYSIS:
 * Kernel: SpMV in CSR format with uniform nonzeros per row (16).
 * Bottleneck: Memory bandwidth (low arithmetic intensity due to random x access).
 * Optimizations applied:
 *   1. NUMA-aware first-touch placement: parallel zero-initialization of arrays
 *      using the same row-block schedule as SpMV to allocate pages on local NUMA node.
 *   2. Preserved existing static schedule (uniform work) and AVX vectorization.
 *   3. No changes to algorithm, SIMD intrinsics, or memory layout.
 */

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <immintrin.h>
#include <omp.h>

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
    #pragma omp parallel for schedule(static) default(none) \
        firstprivate(N) shared(row_ptr, col, val, x, y, b)
    for (long row_start = 0; row_start < N; row_start += BLOCK_ROWS) {
        long row_end = row_start + BLOCK_ROWS;
        if (row_end > N) row_end = N;
        for (long i = row_start; i < row_end; ++i) {
            __m256d acc = _mm256_setzero_pd();          // vector accumulator
            long k = row_ptr[i];
            long k_end = row_ptr[i + 1];
            // main vector loop, width = 4
            for (; k + 3 < k_end; k += 4) {
                // prefetch future x values
                __builtin_prefetch(&x[col[k + PREFETCH_DIST]], 0, 1);
                __builtin_prefetch(&x[col[k + PREFETCH_DIST + 1]], 0, 1);
                __builtin_prefetch(&x[col[k + PREFETCH_DIST + 2]], 0, 1);
                __builtin_prefetch(&x[col[k + PREFETCH_DIST + 3]], 0, 1);

                // load 4 contiguous val entries
                __m256d v_val = _mm256_loadu_pd(&val[k]);

                // gather 4 x entries via col indices
                __m128i v_idx = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&col[k]));
                __m256d v_x = _mm256_i32gather_pd(x, v_idx, 8);   // scale = sizeof(double)=8

                // fused multiply‑add
                acc = _mm256_fmadd_pd(v_val, v_x, acc);
            }
            // scalar tail
            double tail_sum = 0.0;
            for (; k < k_end; ++k) {
                tail_sum += val[k] * x[col[k]];
            }

            // horizontal reduction of acc
            __m128d lo = _mm256_castpd256_pd128(acc);
            __m128d hi = _mm256_extractf128_pd(acc, 1);
            __m128d sum2 = _mm_add_pd(lo, hi);                 // sum pairs
            __m128d shuf = _mm_shuffle_pd(sum2, sum2, 0x1);
            __m128d total_vec = _mm_add_sd(sum2, shuf);
            double row_sum = _mm_cvtsd_f64(total_vec) + tail_sum;

            y[i] = row_sum + b[i];
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

    // NUMA-aware first-touch: zero-initialize arrays with same mapping as kernel
    #pragma omp parallel for schedule(static)
    for (long row_start = 0; row_start < N; row_start += BLOCK_ROWS) {
        long row_end = row_start + BLOCK_ROWS;
        if (row_end > N) row_end = N;

        // Touch row_ptr: from row_start to row_end (inclusive)
        for (long i = row_start; i <= row_end; ++i) {
            row_ptr[i] = 0;
        }

        // Touch col and val: for each row in [row_start, row_end-1]
        for (long i = row_start; i < row_end; ++i) {
            long idx_start = i * NNZ_PER_ROW;
            long idx_end = idx_start + NNZ_PER_ROW;
            for (long j = idx_start; j < idx_end; ++j) {
                col[j] = 0;
                val[j] = 0;
            }
        }

        // Touch b, y, x: by row index
        for (long i = row_start; i < row_end; ++i) {
            b[i] = 0;
            y[i] = 0;
            x[i] = 0;
        }
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

    double start = omp_get_wtime();

    for (int iter = 0; iter < NUM_ITERS; ++iter) {
        spmv_kernel(N, row_ptr, col, val, x, y, b);
        std::swap(x, y);
    }

    double end = omp_get_wtime();
    double elapsed_ms = (end - start) * 1000.0;

    std::printf("Tiempo de ejecucion de la SpMV (%d iters): %.2f ms\n", NUM_ITERS, elapsed_ms);

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