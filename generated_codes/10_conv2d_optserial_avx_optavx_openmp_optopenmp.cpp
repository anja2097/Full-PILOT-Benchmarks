/* ANALYSIS: Conv2D 9x9 stencil, N=4096, 8 iters, AVX2+FMA, TILE_J=56
   Bottleneck: memory bandwidth (9-row reuse ~288KB/L2, streaming rows) + NUMA remote + fork overhead
   - Single parallel region outside iter loop (1 fork vs 8), local cur/nxt swap, for-barrier per iter
   - Removed collapse(2): row-only static partition for NUMA row locality (collapse touched all rows per thread)
   - Parallel first-touch zeros with same row-static mapping before serial rand/memcpy to place pages locally
   - Kept AVX2/FMA kernel and TILE_J blocking unchanged; uniform work keeps schedule(static)
 */
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstddef>
#include <immintrin.h>
#include <omp.h>

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

    // NUMA first-touch with same row-static mapping as timed kernel
    #pragma omp parallel for schedule(static) default(none) shared(in, out)
    for (int i = 0; i < N; ++i) {
        double* rin = in + (size_t)i * N;
        double* rout = out + (size_t)i * N;
        for (int j = 0; j < N; ++j) {
            rin[j] = 0.0;
            rout[j] = 0.0;
        }
    }

    srand(42);
    for (size_t p = 0; p < total; ++p) {
        in[p] = static_cast<double>(rand()) / RAND_MAX;
    }
    memcpy(out, in, total * sizeof(double));

    double start = omp_get_wtime();

    #pragma omp parallel default(none) shared(in, out, filter, offsets)
    {
        double* lcur = in;
        double* lnxt = out;
        for (int iter = 0; iter < NUM_ITERS; ++iter) {
            const double* __restrict__ cin = lcur;
            double* __restrict__ cout = lnxt;

            #pragma omp for schedule(static)
            for (int i = R; i < N - R; ++i) {
                for (int jj = R; jj < N - R; jj += TILE_J) {
                    const double* base = cin + (size_t)i * N + jj;
                    double* outrow = cout + (size_t)i * N + jj;

                    constexpr int VEC_W = 4;
                    constexpr int NB = TILE_J / VEC_W;
                    __m256d acc[NB];

                    {
                        double w0 = filter[0];
                        __m256d w0v = _mm256_set1_pd(w0);
                        const double* row0 = base + offsets[0];
                        for (int b = 0; b < NB; ++b) {
                            __m256d vec = _mm256_loadu_pd(row0 + b * VEC_W);
                            acc[b] = _mm256_mul_pd(w0v, vec);
                        }
                    }

                    for (int k = 1; k < W * W; ++k) {
                        double wk = filter[k];
                        __m256d wkv = _mm256_set1_pd(wk);
                        const double* rowk = base + offsets[k];
                        for (int b = 0; b < NB; ++b) {
                            __m256d vec = _mm256_loadu_pd(rowk + b * VEC_W);
                            acc[b] = _mm256_fmadd_pd(wkv, vec, acc[b]);
                        }
                    }

                    for (int b = 0; b < NB; ++b) {
                        _mm256_storeu_pd(outrow + b * VEC_W, acc[b]);
                    }
                }
            }

            double* tmp = lcur;
            lcur = lnxt;
            lnxt = tmp;
        }
    }

    double end = omp_get_wtime();
    double elapsed = (end - start) * 1000.0; // milliseconds

    printf("Tiempo de ejecucion de la convolucion 2D (%d iters): %.2f ms\n", NUM_ITERS, elapsed);

    double* cur = (NUM_ITERS % 2 == 0) ? in : out;
    double checksum = 0.0;
    for (size_t p = 0; p < total; ++p) {
        checksum += cur[p] * static_cast<double>(p % 11 + 1);
    }
    printf("Checksum: %.6f\n", checksum);

    free(in);
    free(out);
    return 0;
}