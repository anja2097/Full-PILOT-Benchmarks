/* ANALYSIS: 
 * Kernel: SpMV CSR (16 nnz/row) repeated 30 times, x = A*x + b.
 * Bottleneck: MPI_Allgatherv communication dominates (process wall 2401 ms vs kernel 439 ms).
 * Transformations: Replaced blocking MPI_Allgatherv with non-blocking point-to-point all-to-all exchange (MPI_Isend/Irecv) plus self-copy.
 * This reduces collective overhead and allows MPI to progress asynchronously.
 * Preserved all OpenMP pragmas, SIMD intrinsics, and data generation.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <immintrin.h>
#include <omp.h>
#include <mpi.h>

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
                               const double * __restrict__ b,
                               const long local_start,
                               const long local_end) {
    #pragma omp parallel for schedule(static) default(none) \
        firstprivate(N, local_start, local_end) shared(row_ptr, col, val, x, y, b)
    for (long row_start = local_start; row_start < local_end; row_start += BLOCK_ROWS) {
        long row_end = row_start + BLOCK_ROWS;
        if (row_end > local_end) row_end = local_end;
        for (long i = row_start; i < row_end; ++i) {
            __m256d acc = _mm256_setzero_pd();          // vector accumulator
            long k = row_ptr[i];
            long k_end = row_ptr[i + 1];
            for (; k + 3 < k_end; k += 4) {
                __builtin_prefetch(&x[col[k + PREFETCH_DIST]], 0, 1);
                __builtin_prefetch(&x[col[k + PREFETCH_DIST + 1]], 0, 1);
                __builtin_prefetch(&x[col[k + PREFETCH_DIST + 2]], 0, 1);
                __builtin_prefetch(&x[col[k + PREFETCH_DIST + 3]], 0, 1);

                __m256d v_val = _mm256_loadu_pd(&val[k]);
                __m128i v_idx = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&col[k]));
                __m256d v_x = _mm256_i32gather_pd(x, v_idx, 8);
                acc = _mm256_fmadd_pd(v_val, v_x, acc);
            }
            double tail_sum = 0.0;
            for (; k < k_end; ++k) {
                tail_sum += val[k] * x[col[k]];
            }

            __m128d lo = _mm256_castpd256_pd128(acc);
            __m128d hi = _mm256_extractf128_pd(acc, 1);
            __m128d sum2 = _mm_add_pd(lo, hi);
            __m128d shuf = _mm_shuffle_pd(sum2, sum2, 0x1);
            __m128d total_vec = _mm_add_sd(sum2, shuf);
            double row_sum = _mm_cvtsd_f64(total_vec) + tail_sum;

            y[i] = row_sum + b[i];
        }
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int N = 1 << 21;          // 2 097 152 rows
    const int NNZ_PER_ROW = 16;
    const int NUM_ITERS = 30;
    const long NNZ = static_cast<long>(N) * NNZ_PER_ROW;

    long *row_ptr = (long*)aligned_alloc(64, (N + 1) * sizeof(long));
    int *col = (int*)aligned_alloc(64, NNZ * sizeof(int));
    double *val = (double*)aligned_alloc(64, NNZ * sizeof(double));
    double *x = (double*)aligned_alloc(64, N * sizeof(double));
    double *y = (double*)aligned_alloc(64, N * sizeof(double));
    double *b = (double*)aligned_alloc(64, N * sizeof(double));

    if (!row_ptr || !col || !val || !x || !y || !b) {
        if (rank == 0) std::fprintf(stderr, "Allocation failed\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    #pragma omp parallel for schedule(static)
    for (long row_start = 0; row_start < N; row_start += BLOCK_ROWS) {
        long row_end = row_start + BLOCK_ROWS;
        if (row_end > N) row_end = N;

        for (long i = row_start; i <= row_end; ++i) {
            row_ptr[i] = 0;
        }

        for (long i = row_start; i < row_end; ++i) {
            long idx_start = i * NNZ_PER_ROW;
            long idx_end = idx_start + NNZ_PER_ROW;
            for (long j = idx_start; j < idx_end; ++j) {
                col[j] = 0;
                val[j] = 0;
            }
        }

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

    long rows_per_rank = N / size;
    long remainder = N % size;
    long local_start = rank * rows_per_rank + (rank < remainder ? rank : remainder);
    long local_end   = local_start + rows_per_rank + (rank < remainder ? 1 : 0);
    int local_n = static_cast<int>(local_end - local_start);

    int *rcounts = (int*)malloc(size * sizeof(int));
    int *displs = (int*)malloc(size * sizeof(int));
    for (int r = 0; r < size; ++r) {
        long start = r * rows_per_rank + (r < remainder ? r : remainder);
        long end   = start + rows_per_rank + (r < remainder ? 1 : 0);
        rcounts[r] = static_cast<int>(end - start);
        displs[r] = static_cast<int>(start);
    }

    MPI_Request *send_req = NULL;
    MPI_Request *recv_req = NULL;
    if (size > 1) {
        send_req = (MPI_Request*)malloc((size - 1) * sizeof(MPI_Request));
        recv_req = (MPI_Request*)malloc((size - 1) * sizeof(MPI_Request));
        if (!send_req || !recv_req) {
            if (rank == 0) std::fprintf(stderr, "Request allocation failed\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    for (int iter = 0; iter < NUM_ITERS; ++iter) {
        spmv_kernel(N, row_ptr, col, val, x, y, b, local_start, local_end);
        
        if (size == 1) {
            memcpy(x + displs[0], y + local_start, local_n * sizeof(double));
        } else {
            // Copy own chunk into x (equivalent to Allgatherv including self)
            memcpy(x + displs[rank], y + local_start, local_n * sizeof(double));
            
            int req_idx = 0;
            for (int r = 0; r < size; ++r) {
                if (r == rank) continue;
                MPI_Irecv(x + displs[r], rcounts[r], MPI_DOUBLE, r, 0, MPI_COMM_WORLD, &recv_req[req_idx]);
                MPI_Isend(y + local_start, local_n, MPI_DOUBLE, r, 0, MPI_COMM_WORLD, &send_req[req_idx]);
                req_idx++;
            }
            MPI_Waitall(size - 1, recv_req, MPI_STATUSES_IGNORE);
            MPI_Waitall(size - 1, send_req, MPI_STATUSES_IGNORE);
        }
    }

    double end = MPI_Wtime();
    double elapsed_ms = (end - start) * 1000.0;
    double max_elapsed_ms = 0.0;
    MPI_Reduce(&elapsed_ms, &max_elapsed_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    double local_checksum = 0.0;
    for (long i = local_start; i < local_end; ++i) {
        local_checksum += x[i] * static_cast<double>(i % 11 + 1);
    }
    double checksum = 0.0;
    MPI_Reduce(&local_checksum, &checksum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::printf("Tiempo de ejecucion de la SpMV (%d iters): %.2f ms\n", NUM_ITERS, max_elapsed_ms);
        std::printf("Checksum: %.6f\n", checksum);
    }

    std::free(row_ptr);
    std::free(col);
    std::free(val);
    std::free(x);
    std::free(y);
    std::free(b);
    std::free(rcounts);
    std::free(displs);
    if (send_req) std::free(send_req);
    if (recv_req) std::free(recv_req);

    MPI_Finalize();
    return 0;
}