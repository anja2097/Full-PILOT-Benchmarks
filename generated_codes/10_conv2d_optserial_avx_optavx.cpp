/* ANALYSIS: AVX2 vectorized 2D convolution inner j-loop, compute-bound (5x5 filter), using __m256d, FMA, unaligned loads/stores, register blocking 14 vectors (56 columns). */
/* original includes */
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstddef>
#include <immintrin.h>

#define TILE_J 56
#define N 4096
#define R 4
#define W (2 * R + 1)
#define NUM_ITERS 8

int main()
{
    // Compute filter
    double filter[W * W] __attribute__((aligned(64)));
    double fsum = 0.0;
    for (int dy = -R; dy <= R; ++dy) {
        for (int dx = -R; dx <= R; ++dx) {
            double w = std::exp(-(dx * dx + dy * dy) / 8.0);
            filter[(dy + R) * W + (dx + R)] = w;
            fsum += w;
        }
    }
    for (int k = 0; k < W * W; ++k) filter[k] /= fsum;

    // Precompute offsets for each tap
    ptrdiff_t offsets[W * W];
    for (int dy = -R; dy <= R; ++dy) {
        for (int dx = -R; dx <= R; ++dx) {
            int k = (dy + R) * W + (dx + R);
            offsets[k] = (ptrdiff_t)dy * N + dx;
        }
    }

    size_t total = (size_t)N * N;
    double* in = (double*)aligned_alloc(64, total * sizeof(double));
    double* out = (double*)aligned_alloc(64, total * sizeof(double));
    if (!in || !out) return 1;

    srand(42);
    for (size_t p = 0; p < total; ++p) {
        in[p] = static_cast<double>(rand()) / RAND_MAX;
    }
    memcpy(out, in, total * sizeof(double));

    auto start = std::chrono::high_resolution_clock::now();

    double* cur = in;
    double* nxt = out;

    for (int iter = 0; iter < NUM_ITERS; ++iter) {
        const double* __restrict__ cin = cur;
        double* __restrict__ cout = nxt;

        for (int jj = R; jj < N - R; jj += TILE_J) {
            for (int i = R; i < N - R; ++i) {
                const double* base = cin + (size_t)i * N + jj;
                double* outrow = cout + (size_t)i * N + jj;

                constexpr int VEC_W = 4;                     // AVX2 width for double
                constexpr int NB = TILE_J / VEC_W;           // 56/4 = 14
                __m256d acc[NB];

                // First tap: initialize accumulators
                {
                    double w0 = filter[0];
                    __m256d w0v = _mm256_set1_pd(w0);
                    const double* row0 = base + offsets[0];
                    for (int b = 0; b < NB; ++b) {
                        __m256d vec = _mm256_loadu_pd(row0 + b * VEC_W);
                        acc[b] = _mm256_mul_pd(w0v, vec);
                    }
                }

                // Remaining taps: accumulate
                for (int k = 1; k < W * W; ++k) {
                    double wk = filter[k];
                    __m256d wkv = _mm256_set1_pd(wk);
                    const double* rowk = base + offsets[k];
                    for (int b = 0; b < NB; ++b) {
                        __m256d vec = _mm256_loadu_pd(rowk + b * VEC_W);
                        acc[b] = _mm256_fmadd_pd(wkv, vec, acc[b]);
                    }
                }

                // Store result
                for (int b = 0; b < NB; ++b) {
                    _mm256_storeu_pd(outrow + b * VEC_W, acc[b]);
                }
            }
        }

        double* tmp = cur;
        cur = nxt;
        nxt = tmp;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    printf("Tiempo de ejecucion de la convolucion 2D (%d iters): %.2f ms\n", NUM_ITERS, elapsed.count());

    double checksum = 0.0;
    for (size_t p = 0; p < total; ++p) {
        checksum += cur[p] * static_cast<double>(p % 11 + 1);
    }
    printf("Checksum: %.6f\n", checksum);

    free(in);
    free(out);
    return 0;
}