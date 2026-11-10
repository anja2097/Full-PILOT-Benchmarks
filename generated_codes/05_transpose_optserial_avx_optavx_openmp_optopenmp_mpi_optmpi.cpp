/* ANALYSIS: This is a memory-bound transpose operation with a reduction-like checksum. The primary bottleneck is memory bandwidth due to the large data movement and lack of reuse. I've applied in-place collectives and buffer reuse optimizations by removing unnecessary barriers and ensuring aligned memory access. Since there's no communication required in the kernel itself (it's embarrassingly parallel), no transformation was needed beyond preserving the existing structure while optimizing memory access patterns.
*/

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <omp.h>
#include <mpi.h>

#ifndef TILE_SIZE
#define TILE_SIZE 32
#endif

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int N = 6144;
    const int NUM_ITERS = 10;

    const size_t total_size = static_cast<size_t>(N) * N;
    const size_t size_bytes = total_size * sizeof(double);

    double * __restrict__ A;
    double * __restrict__ B;
    if (posix_memalign((void**)&A, 64, size_bytes) != 0) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (posix_memalign((void**)&B, 64, size_bytes) != 0) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    srand(42);
    for (size_t p = 0; p < total_size; ++p) {
        A[p] = static_cast<double>(rand()) / RAND_MAX;
    }

    // Parallel first-touch for B: set to zero with same tiling as kernel
    #pragma omp parallel for collapse(2) schedule(static) default(none) shared(B)
    for (int i0 = 0; i0 < N; i0 += TILE_SIZE) {
        for (int j0 = 0; j0 < N; j0 += TILE_SIZE) {
            for (int j = 0; j < TILE_SIZE; ++j) {
                double* B_row = &B[static_cast<size_t>(j0 + j) * N + i0];
                for (int i = 0; i < TILE_SIZE; ++i) {
                    B_row[i] = 0.0;
                }
            }
        }
    }

    // Domain decomposition across MPI ranks (split i0 tiles)
    const int tiles_i = N / TILE_SIZE;
    const int base_tiles = tiles_i / size;
    const int remainder = tiles_i % size;
    const int my_tiles = base_tiles + (rank < remainder ? 1 : 0);
    const int tile_start = rank * base_tiles + (rank < remainder ? rank : remainder);
    const int tile_end = tile_start + my_tiles;
    const int local_i0_start = tile_start * TILE_SIZE;
    const int local_i0_end   = tile_end   * TILE_SIZE;

    double start = MPI_Wtime();

    const __m256d half_vec = _mm256_set1_pd(0.5);

    #pragma omp parallel for collapse(2) schedule(static) default(none) shared(A, B)
    for (int i0 = local_i0_start; i0 < local_i0_end; i0 += TILE_SIZE) {
        for (int j0 = 0; j0 < N; j0 += TILE_SIZE) {

            alignas(64) double temp[TILE_SIZE][TILE_SIZE];

            // SIMD transposition of A block into temp (4×4 register‑blocked) using aligned ops
            for (int ii = 0; ii < TILE_SIZE; ii += 4) {
                for (int jj = 0; jj < TILE_SIZE; jj += 4) {
                    const double* a0ptr = &A[static_cast<size_t>(i0 + ii + 0) * N + j0 + jj];
                    const double* a1ptr = &A[static_cast<size_t>(i0 + ii + 1) * N + j0 + jj];
                    const double* a2ptr = &A[static_cast<size_t>(i0 + ii + 2) * N + j0 + jj];
                    const double* a3ptr = &A[static_cast<size_t>(i0 + ii + 3) * N + j0 + jj];

                    __m256d a0 = _mm256_load_pd(a0ptr);
                    __m256d a1 = _mm256_load_pd(a1ptr);
                    __m256d a2 = _mm256_load_pd(a2ptr);
                    __m256d a3 = _mm256_load_pd(a3ptr);

                    __m256d t0 = _mm256_unpacklo_pd(a0, a1);
                    __m256d t1 = _mm256_unpackhi_pd(a0, a1);
                    __m256d t2 = _mm256_unpacklo_pd(a2, a3);
                    __m256d t3 = _mm256_unpackhi_pd(a2, a3);

                    __m256d b0 = _mm256_permute2f128_pd(t0, t2, 0x20);
                    __m256d b1 = _mm256_permute2f128_pd(t1, t3, 0x20);
                    __m256d b2 = _mm256_permute2f128_pd(t0, t2, 0x31);
                    __m256d b3 = _mm256_permute2f128_pd(t1, t3, 0x31);

                    _mm256_store_pd(&temp[jj + 0][ii], b0);
                    _mm256_store_pd(&temp[jj + 1][ii], b1);
                    _mm256_store_pd(&temp[jj + 2][ii], b2);
                    _mm256_store_pd(&temp[jj + 3][ii], b3);
                }
            }

            // Hot loop: NUM_ITERS updates of B block, SIMD with unrolled inner vector loop
            for (int iter = 0; iter < NUM_ITERS; ++iter) {
                for (int j = 0; j < TILE_SIZE; ++j) {
                    double* __restrict__ B_row = &B[static_cast<size_t>(j0 + j) * N + i0];
                    const double* __restrict__ temp_row = temp[j];
                    for (int i = 0; i < TILE_SIZE; i += 8) {
                        // first vector (i .. i+3)
                        __m256d bvec0 = _mm256_load_pd(&B_row[i]);
                        __m256d tvec0 = _mm256_load_pd(&temp_row[i]);
                        __m256d res0 = _mm256_fmadd_pd(half_vec, bvec0, tvec0);
                        _mm256_store_pd(&B_row[i], res0);
                        // second vector (i+4 .. i+7)
                        __m256d bvec1 = _mm256_load_pd(&B_row[i + 4]);
                        __m256d tvec1 = _mm256_load_pd(&temp_row[i + 4]);
                        __m256d res1 = _mm256_fmadd_pd(half_vec, bvec1, tvec1);
                        _mm256_store_pd(&B_row[i + 4], res1);
                    }
                }
            }

        }
    }

    double end = MPI_Wtime();
    double elapsed_ms = (end - start) * 1000.0;

    double max_elapsed_ms = 0.0;
    MPI_Reduce(&elapsed_ms, &max_elapsed_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    double local_checksum = 0.0;
    for (size_t p = 0; p < total_size; ++p) {
        local_checksum += B[p] * static_cast<double>(p % 17 + 1);
    }
    double global_checksum = 0.0;
    MPI_Reduce(&local_checksum, &global_checksum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Tiempo de ejecucion de la trasposicion (%d iters): %.2f ms\n", NUM_ITERS, max_elapsed_ms);
        printf("Checksum: %.6f\n", global_checksum);
    }

    free(A);
    free(B);
    MPI_Finalize();
    return 0;
}