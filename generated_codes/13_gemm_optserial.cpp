/* ANALYSIS:
 * Kernel: dense GEMM C+=A*B with C*=0.5 per iter (M=1200,N=1000,K=1100).
 * Bottleneck: memory latency/bandwidth - naive i-j-k inner-k strides B by N.
 * Transform: scale C by 0.5 streaming pass + cache-blocked i0-j0-k0,i-k-j axpy.
 * Inner jj contiguous on B/C store for auto-vectorization; C stationary in L2.
 * Tiles sized for L1d 32K/L2 512K; multiples of 4 doubles for AVX2; restrict+64B align.
 */
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <algorithm>

#ifndef TILE_M
#define TILE_M 96
#endif
#ifndef TILE_N
#define TILE_N 128
#endif
#ifndef TILE_K
#define TILE_K 128
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
        for (size_t p = 0; p < sizeC; ++p) {
            Cr[p] *= 0.5;
        }

        for (int i0 = 0; i0 < M; i0 += TILE_M) {
            int i1 = i0 + TILE_M < M ? i0 + TILE_M : M;
            for (int j0 = 0; j0 < N; j0 += TILE_N) {
                int j1 = j0 + TILE_N < N ? j0 + TILE_N : N;
                int nj = j1 - j0;
                for (int k0 = 0; k0 < K; k0 += TILE_K) {
                    int k1 = k0 + TILE_K < K ? k0 + TILE_K : K;
                    for (int i = i0; i < i1; ++i) {
                        double * __restrict__ Crow = Cr + static_cast<size_t>(i) * N + j0;
                        const double * __restrict__ Arow = Ar + static_cast<size_t>(i) * K;
                        for (int k = k0; k < k1; ++k) {
                            double aval = Arow[k];
                            const double * __restrict__ Brow = Br + static_cast<size_t>(k) * N + j0;
                            for (int jj = 0; jj < nj; ++jj) {
                                Crow[jj] += aval * Brow[jj];
                            }
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