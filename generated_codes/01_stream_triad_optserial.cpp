/* ANALYSIS: STREAM Triad kernel, memory‑bandwidth bound. Applied temporal tiling (TILE_SIZE) so b,c stay in L2 across 100 iterations, aligned all arrays to 64 B, added __restrict__ for aliasing. Improves cache reuse and helps auto‑vectorization. */
#include <chrono>
#include <cstdio>
#include <cstdlib>

#ifndef TILE_SIZE
#define TILE_SIZE 8192          // elements per block, fits in L2 (≈64 KB per array)
#endif

int main() {
    const long N = 1L << 25;          // 33 554 432 elements
    const int NUM_ITERS = 100;
    const double s = 0.75;

    double * __restrict__ a;
    double * __restrict__ b;
    double * __restrict__ c;

    if (posix_memalign((void **)&a, 64, N * sizeof(double)) != 0 ||
        posix_memalign((void **)&b, 64, N * sizeof(double)) != 0 ||
        posix_memalign((void **)&c, 64, N * sizeof(double)) != 0) {
        std::fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    std::srand(42);
    for (long i = 0; i < N; ++i) a[i] = static_cast<double>(std::rand()) / RAND_MAX;
    for (long i = 0; i < N; ++i) b[i] = static_cast<double>(std::rand()) / RAND_MAX;
    for (long i = 0; i < N; ++i) c[i] = static_cast<double>(std::rand()) / RAND_MAX;

    auto start = std::chrono::high_resolution_clock::now();

    for (long tile = 0; tile < N; tile += TILE_SIZE) {
        const long tile_end = tile + TILE_SIZE;
        for (int iter = 0; iter < NUM_ITERS; ++iter) {
            for (long i = tile; i < tile_end; ++i) {
                a[i] = 0.5 * a[i] + b[i] + s * c[i];
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::printf("Tiempo de ejecucion de la triada (%d iters): %.2f ms\n", NUM_ITERS, elapsed.count());

    double checksum = 0.0;
    for (long i = 0; i < N; ++i) {
        checksum += a[i] * static_cast<double>((i % 13) + 1);
    }
    std::printf("Checksum: %.6f\n", checksum);

    std::free(a);
    std::free(b);
    std::free(c);
    return 0;
}