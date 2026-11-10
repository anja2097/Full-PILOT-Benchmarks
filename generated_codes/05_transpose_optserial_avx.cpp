/* ANALYSIS: AVX2 SIMD, compute‑bound (multiply‑add & transpose), vectorized inner tile loops with register‑blocked 4×4 transpose and fused multiply‑add for B update. */
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>

#ifndef TILE_SIZE
#define TILE_SIZE 32
#endif

int main()
{
    const int N = 6144;
    const int NUM_ITERS = 10;

    const size_t total_size = static_cast<size_t>(N) * N;
    const size_t size_bytes = total_size * sizeof(double);

    double * __restrict__ A;
    double * __restrict__ B;
    if (posix_memalign((void**)&A, 64, size_bytes) != 0) return 1;
    if (posix_memalign((void**)&B, 64, size_bytes) != 0) return 1;

    memset(B, 0, size_bytes);

    srand(42);
    for (size_t p = 0; p < total_size; ++p) {
        A[p] = static_cast<double>(rand()) / RAND_MAX;
    }

    auto start = std::chrono::high_resolution_clock::now();

    alignas(64) double temp[TILE_SIZE][TILE_SIZE];

    const __m256d half_vec = _mm256_set1_pd(0.5);

    for (int i0 = 0; i0 < N; i0 += TILE_SIZE) {
        for (int j0 = 0; j0 < N; j0 += TILE_SIZE) {

            // SIMD transposition of A block into temp (4×4 register‑blocked)
            for (int ii = 0; ii < TILE_SIZE; ii += 4) {
                for (int jj = 0; jj < TILE_SIZE; jj += 4) {
                    // Load four rows of A, each four contiguous doubles
                    const double* a0ptr = &A[static_cast<size_t>(i0 + ii + 0) * N + j0 + jj];
                    const double* a1ptr = &A[static_cast<size_t>(i0 + ii + 1) * N + j0 + jj];
                    const double* a2ptr = &A[static_cast<size_t>(i0 + ii + 2) * N + j0 + jj];
                    const double* a3ptr = &A[static_cast<size_t>(i0 + ii + 3) * N + j0 + jj];

                    __m256d a0 = _mm256_loadu_pd(a0ptr);
                    __m256d a1 = _mm256_loadu_pd(a1ptr);
                    __m256d a2 = _mm256_loadu_pd(a2ptr);
                    __m256d a3 = _mm256_loadu_pd(a3ptr);

                    // Transpose the 4×4 matrix
                    __m256d t0 = _mm256_unpacklo_pd(a0, a1); // a0[0],a1[0], a0[1],a1[1]
                    __m256d t1 = _mm256_unpackhi_pd(a0, a1); // a0[2],a1[2], a0[3],a1[3]
                    __m256d t2 = _mm256_unpacklo_pd(a2, a3);
                    __m256d t3 = _mm256_unpackhi_pd(a2, a3);

                    __m256d b0 = _mm256_permute2f128_pd(t0, t2, 0x20);
                    __m256d b1 = _mm256_permute2f128_pd(t1, t3, 0x20);
                    __m256d b2 = _mm256_permute2f128_pd(t0, t2, 0x31);
                    __m256d b3 = _mm256_permute2f128_pd(t1, t3, 0x31);

                    // Store transposed vectors into temp
                    _mm256_storeu_pd(&temp[jj + 0][ii], b0);
                    _mm256_storeu_pd(&temp[jj + 1][ii], b1);
                    _mm256_storeu_pd(&temp[jj + 2][ii], b2);
                    _mm256_storeu_pd(&temp[jj + 3][ii], b3);
                }
            }

            // Hot loop: 10 iterations over B block, SIMD update
            for (int iter = 0; iter < NUM_ITERS; ++iter) {
                for (int j = 0; j < TILE_SIZE; ++j) {
                    double* __restrict__ B_row = &B[static_cast<size_t>(j0 + j) * N + i0];
                    const double* __restrict__ temp_row = temp[j];
                    for (int i = 0; i < TILE_SIZE; i += 4) {
                        __m256d bvec = _mm256_loadu_pd(&B_row[i]);
                        __m256d tvec = _mm256_loadu_pd(&temp_row[i]);
                        __m256d res = _mm256_fmadd_pd(half_vec, bvec, tvec); // 0.5*B + temp
                        _mm256_storeu_pd(&B_row[i], res);
                    }
                }
            }

        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    printf("Tiempo de ejecucion de la trasposicion (%d iters): %.2f ms\n", NUM_ITERS, elapsed.count());

    double checksum = 0.0;
    for (size_t p = 0; p < total_size; ++p) {
        checksum += B[p] * static_cast<double>(p % 17 + 1);
    }
    printf("Checksum: %.6f\n", checksum);

    free(A);
    free(B);
    return 0;
}