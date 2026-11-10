/* ANALYSIS: 
   Kernel: 2D 9x9 Gaussian convolution, 8 iterations, double buffer.
   Bottleneck: serial dependency chain and poor cache reuse in naive loop.
   Transformations:
   - Tiled column loop (TILE_J=56) to keep working set in L1d (9 rows * 64 cols = 4.6KB).
   - Loop interchange: accumulate over kernel taps for a whole row tile, making the innermost loop unit-stride over columns.
   - Precomputed kernel offsets and aligned temporary buffer.
   - 64-byte aligned heap arrays, __restrict__ pointers.
*/

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstddef>

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
                double out_temp[TILE_J] __attribute__((aligned(64)));
                const double* base = cin + (size_t)i * N + jj;

                // First tap: initialize
                {
                    double w = filter[0];
                    const double* row = base + offsets[0];
                    for (int j = 0; j < TILE_J; ++j) {
                        out_temp[j] = w * row[j];
                    }
                }
                // Remaining taps: accumulate
                for (int k = 1; k < W * W; ++k) {
                    double w = filter[k];
                    const double* row = base + offsets[k];
                    for (int j = 0; j < TILE_J; ++j) {
                        out_temp[j] += w * row[j];
                    }
                }
                // Store result
                double* outrow = cout + (size_t)i * N + jj;
                for (int j = 0; j < TILE_J; ++j) {
                    outrow[j] = out_temp[j];
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