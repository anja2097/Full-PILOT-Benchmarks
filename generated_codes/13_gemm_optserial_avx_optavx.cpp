/* ANALYSIS: 
 * Kernel: GEMM (C = 0.5*C + A*B) over 5 iterations.
 * Bottleneck: memory bandwidth due to poor data reuse (B read M times).
 * Transformations: 
 *   - Reordered loops to j0, k0, i0 to reuse B across all M and A across N.
 *   - Register blocking MR=4, NR=8 with AVX2 FMA.
 *   - Cache blocking MC=128, NC=128, KC=256 to fit A and B blocks in L2.
 *   - Used aligned loads/stores (64-byte aligned arrays).
 *   - Unrolled k loop by 2 for better FMA throughput.
 */
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <immintrin.h>

#ifndef MR
#define MR 4
#endif
#ifndef NR
#define NR 8
#endif
#ifndef MC
#define MC 128
#endif
#ifndef NC
#define NC 128
#endif
#ifndef KC
#define KC 256
#endif

int main()
{
    const int M = 1200;
    const int N = 1000;
    const int K = 1100;
    const int NUM_ITERS = 5;

    double *A = nullptr;
    double *B = nullptr;
    double *C = nullptr;
    size_t sizeA = static_cast<size_t>(M) * K;
    size_t sizeB = static_cast<size_t>(K) * N;
    size_t sizeC = static_cast<size_t>(M) * N;
    if (posix_memalign(reinterpret_cast<void**>(&A), 64, sizeA * sizeof(double)) != 0) return 1;
    if (posix_memalign(reinterpret_cast<void**>(&B), 64, sizeB * sizeof(double)) != 0) return 1;
    if (posix_memalign(reinterpret_cast<void**>(&C), 64, sizeC * sizeof(double)) != 0) return 1;

    srand(42);
    for (size_t p = 0; p < sizeA; ++p) A[p] = static_cast<double>(rand()) / RAND_MAX;
    for (size_t p = 0; p < sizeB; ++p) B[p] = static_cast<double>(rand()) / RAND_MAX;
    for (size_t p = 0; p < sizeC; ++p) C[p] = 0.0;

    double * __restrict__ Ar = A;
    double * __restrict__ Br = B;
    double * __restrict__ Cr = C;

    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < NUM_ITERS; ++iter) {
        // scale C by 0.5 (vectorized AVX2)
        {
            size_t p = 0;
            const __m256d half = _mm256_set1_pd(0.5);
            for (; p + 4 <= sizeC; p += 4) {
                __m256d v = _mm256_load_pd(Cr + p);
                v = _mm256_mul_pd(v, half);
                _mm256_store_pd(Cr + p, v);
            }
        }

        // GEMM: C += A * B
        for (int j0 = 0; j0 < N; j0 += NC) {
            int j1 = j0 + NC < N ? j0 + NC : N;
            for (int k0 = 0; k0 < K; k0 += KC) {
                int k1 = k0 + KC < K ? k0 + KC : K;
                for (int i0 = 0; i0 < M; i0 += MC) {
                    int i1 = i0 + MC < M ? i0 + MC : M;
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

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    printf("Tiempo de ejecucion de la GEMM (%d iters): %.2f ms\n", NUM_ITERS, elapsed.count());

    double checksum = 0.0;
    for (size_t p = 0; p < sizeC; ++p) checksum += Cr[p] * static_cast<double>(p % 13 + 1);
    printf("Checksum: %.6f\n", checksum);
    free(A);
    free(B);
    free(C);
    return 0;
}