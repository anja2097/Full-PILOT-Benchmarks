#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <immintrin.h>
#include <omp.h>
#include <mpi.h>

#ifndef MR
#define MR 4
#endif
#ifndef NR
#define NR 8
#endif
#ifndef MC
#define MC 60
#endif
#ifndef NC
#define NC 80
#endif
#ifndef KC
#define KC 256
#endif

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int M = 1200;
    const int N = 1000;
    const int K = 1100;
    const int NUM_ITERS = 5;

    /* Determine local row range for each rank */
    int rows_per_rank = M / size;
    int remainder = M % size;
    int local_M = (rank < remainder) ? rows_per_rank + 1 : rows_per_rank;
    int row_start = rank * rows_per_rank + std::min(rank, remainder);
    int row_end = row_start + local_M; /* exclusive */

    size_t sizeA = static_cast<size_t>(M) * K;
    size_t sizeB = static_cast<size_t>(K) * N;
    size_t sizeC_local = static_cast<size_t>(local_M) * N;

    double *A = nullptr;
    double *B = nullptr;
    double *C_local = nullptr;
    if (posix_memalign(reinterpret_cast<void**>(&A), 64, sizeA * sizeof(double)) != 0) return 1;
    if (posix_memalign(reinterpret_cast<void**>(&B), 64, sizeB * sizeof(double)) != 0) return 1;
    if (posix_memalign(reinterpret_cast<void**>(&C_local), 64, sizeC_local * sizeof(double)) != 0) return 1;

    /* First-touch initialization */
    #pragma omp parallel default(none) shared(A,B,C_local,sizeA,sizeB,sizeC_local)
    {
        #pragma omp for schedule(static)
        for (size_t p = 0; p < sizeA; ++p) A[p] = 0.0;
        #pragma omp for schedule(static)
        for (size_t p = 0; p < sizeB; ++p) B[p] = 0.0;
        #pragma omp for schedule(static)
        for (size_t p = 0; p < sizeC_local; ++p) C_local[p] = 0.0;
    }

    /* Pseudo‑random data generation (identical on every rank) */
    srand(42);
    for (size_t p = 0; p < sizeA; ++p) A[p] = static_cast<double>(rand()) / RAND_MAX;
    for (size_t p = 0; p < sizeB; ++p) B[p] = static_cast<double>(rand()) / RAND_MAX;
    /* C_local already zeroed */

    double * __restrict__ Ar_full = A;
    double * __restrict__ Br = B;
    double * __restrict__ Cr = C_local;

    /* Pointer to the slice of A that belongs to this rank */
    double * __restrict__ Ar = Ar_full + static_cast<size_t>(row_start) * K;

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    #pragma omp parallel default(none) shared(Ar, Br, Cr, local_M, row_start, row_end) firstprivate(N, K)
    {
        for (int iter = 0; iter < NUM_ITERS; ++iter) {
            /* scale C by 0.5 (vectorized AVX2) */
            {
                size_t total = static_cast<size_t>(local_M) * static_cast<size_t>(N);
                size_t limit_simd = total - (total % 4);

                #pragma omp for schedule(static,1024)
                for (size_t p = 0; p < limit_simd; p += 4) {
                    const __m256d half = _mm256_set1_pd(0.5);
                    __m256d v = _mm256_load_pd(Cr + p);
                    v = _mm256_mul_pd(v, half);
                    _mm256_store_pd(Cr + p, v);
                }

                #pragma omp for schedule(static)
                for (size_t p = limit_simd; p < total; ++p) {
                    Cr[p] *= 0.5;
                }
            }

            /* GEMM: C += A * B */
            #pragma omp for collapse(2) schedule(static)
            for (int i0 = 0; i0 < local_M; i0 += MC) {
                for (int j0 = 0; j0 < N; j0 += NC) {
                    int j1 = j0 + NC < N ? j0 + NC : N;
                    int i1 = i0 + MC < local_M ? i0 + MC : local_M;
                    for (int k0 = 0; k0 < K; k0 += KC) {
                        int k1 = k0 + KC < K ? k0 + KC : K;
                        for (int i = i0; i < i1; i += MR) {
                            for (int j = j0; j < j1; j += NR) {
                                double* C_row0 = Cr + static_cast<size_t>(i) * N + j;
                                double* C_row1 = C_row0 + N;
                                double* C_row2 = C_row1 + N;
                                double* C_row3 = C_row2 + N;

                                __m256d c00 = _mm256_load_pd(C_row0);
                                __m256d c01 = _mm256_load_pd(C_row0 + 4);
                                __m256d c10 = _mm256_load_pd(C_row1);
                                __m256d c11 = _mm256_load_pd(C_row1 + 4);
                                __m256d c20 = _mm256_load_pd(C_row2);
                                __m256d c21 = _mm256_load_pd(C_row2 + 4);
                                __m256d c30 = _mm256_load_pd(C_row3);
                                __m256d c31 = _mm256_load_pd(C_row3 + 4);

                                const double* A_row0 = Ar + static_cast<size_t>(i) * K;
                                const double* A_row1 = A_row0 + K;
                                const double* A_row2 = A_row1 + K;
                                const double* A_row3 = A_row2 + K;

                                for (int k = k0; k < k1; k += 2) {
                                    const double* B_row0 = Br + static_cast<size_t>(k) * N + j;
                                    const double* B_row1 = Br + static_cast<size_t>(k+1) * N + j;

                                    __m256d b0_0 = _mm256_load_pd(B_row0);
                                    __m256d b1_0 = _mm256_load_pd(B_row0 + 4);
                                    __m256d b0_1 = _mm256_load_pd(B_row1);
                                    __m256d b1_1 = _mm256_load_pd(B_row1 + 4);

                                    __m256d a0v_0 = _mm256_broadcast_sd(A_row0 + k);
                                    __m256d a1v_0 = _mm256_broadcast_sd(A_row1 + k);
                                    __m256d a2v_0 = _mm256_broadcast_sd(A_row2 + k);
                                    __m256d a3v_0 = _mm256_broadcast_sd(A_row3 + k);

                                    __m256d a0v_1 = _mm256_broadcast_sd(A_row0 + k + 1);
                                    __m256d a1v_1 = _mm256_broadcast_sd(A_row1 + k + 1);
                                    __m256d a2v_1 = _mm256_broadcast_sd(A_row2 + k + 1);
                                    __m256d a3v_1 = _mm256_broadcast_sd(A_row3 + k + 1);

                                    c00 = _mm256_fmadd_pd(a0v_0, b0_0, c00);
                                    c01 = _mm256_fmadd_pd(a0v_0, b1_0, c01);
                                    c10 = _mm256_fmadd_pd(a1v_0, b0_0, c10);
                                    c11 = _mm256_fmadd_pd(a1v_0, b1_0, c11);
                                    c20 = _mm256_fmadd_pd(a2v_0, b0_0, c20);
                                    c21 = _mm256_fmadd_pd(a2v_0, b1_0, c21);
                                    c30 = _mm256_fmadd_pd(a3v_0, b0_0, c30);
                                    c31 = _mm256_fmadd_pd(a3v_0, b1_0, c31);

                                    c00 = _mm256_fmadd_pd(a0v_1, b0_1, c00);
                                    c01 = _mm256_fmadd_pd(a0v_1, b1_1, c01);
                                    c10 = _mm256_fmadd_pd(a1v_1, b0_1, c10);
                                    c11 = _mm256_fmadd_pd(a1v_1, b1_1, c11);
                                    c20 = _mm256_fmadd_pd(a2v_1, b0_1, c20);
                                    c21 = _mm256_fmadd_pd(a2v_1, b1_1, c21);
                                    c30 = _mm256_fmadd_pd(a3v_1, b0_1, c30);
                                    c31 = _mm256_fmadd_pd(a3v_1, b1_1, c31);
                                }

                                _mm256_store_pd(C_row0, c00);
                                _mm256_store_pd(C_row0 + 4, c01);
                                _mm256_store_pd(C_row1, c10);
                                _mm256_store_pd(C_row1 + 4, c11);
                                _mm256_store_pd(C_row2, c20);
                                _mm256_store_pd(C_row2 + 4, c21);
                                _mm256_store_pd(C_row3, c30);
                                _mm256_store_pd(C_row3 + 4, c31);
                            }
                        }
                    }
                }
            }
        }
    } /* end parallel region */

    double end = MPI_Wtime();
    double elapsed_ms = (end - start) * 1000.0;

    double max_elapsed_ms = 0.0;
    MPI_Reduce(&elapsed_ms, &max_elapsed_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Tiempo de ejecucion de la GEMM (%d iters): %.2f ms\n", NUM_ITERS, max_elapsed_ms);
    }

    /* Compute local checksum */
    double local_checksum = 0.0;
    for (int i = 0; i < local_M; ++i) {
        int global_i = row_start + i;
        for (int j = 0; j < N; ++j) {
            size_t p = static_cast<size_t>(global_i) * N + j;
            local_checksum += Cr[static_cast<size_t>(i) * N + j] * static_cast<double>(p % 13 + 1);
        }
    }

    double checksum = 0.0;
    MPI_Reduce(&local_checksum, &checksum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Checksum: %.6f\n", checksum);
    }

    free(A);
    free(B);
    free(C_local);
    MPI_Finalize();
    return 0;
}