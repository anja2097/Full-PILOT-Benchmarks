/* ANALYSIS:
 * Kernel: STREAM triad with 100 repeated updates per element (a = 0.5*a + b + s*c).
 * Original bottleneck: L2 bandwidth bound due to re-reading tiles 100 times.
 * Transformation: Interchanged loops so the 100 iterations run in registers.
 * Each element's update sequence is unchanged; only loop order changed.
 * Loaded a,b,c once per 4-vector block, kept b,c in YMM registers, updated a 100 times.
 * Processed 4 AVX2 vectors (16 doubles) per outer iteration for ILP.
 * No cache tiling needed; memory traffic reduced to one read of a,b,c and one write of a.
 * Compute-bound now; AVX2 FMA path preserved.
 */

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <immintrin.h>

int main() {
    const long N = 1L << 25;
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

    const __m256d half_vec = _mm256_set1_pd(0.5);
    const __m256d s_vec    = _mm256_set1_pd(s);

    long i = 0;
    // Process 16 elements (4 AVX2 vectors) per outer iteration
    for (; i + 15 < N; i += 16) {
        __m256d a0 = _mm256_load_pd(a + i);
        __m256d a1 = _mm256_load_pd(a + i + 4);
        __m256d a2 = _mm256_load_pd(a + i + 8);
        __m256d a3 = _mm256_load_pd(a + i + 12);
        __m256d b0 = _mm256_load_pd(b + i);
        __m256d b1 = _mm256_load_pd(b + i + 4);
        __m256d b2 = _mm256_load_pd(b + i + 8);
        __m256d b3 = _mm256_load_pd(b + i + 12);
        __m256d c0 = _mm256_load_pd(c + i);
        __m256d c1 = _mm256_load_pd(c + i + 4);
        __m256d c2 = _mm256_load_pd(c + i + 8);
        __m256d c3 = _mm256_load_pd(c + i + 12);

        for (int iter = 0; iter < NUM_ITERS; ++iter) {
            a0 = _mm256_fmadd_pd(half_vec, a0, b0);
            a0 = _mm256_fmadd_pd(s_vec, c0, a0);
            a1 = _mm256_fmadd_pd(half_vec, a1, b1);
            a1 = _mm256_fmadd_pd(s_vec, c1, a1);
            a2 = _mm256_fmadd_pd(half_vec, a2, b2);
            a2 = _mm256_fmadd_pd(s_vec, c2, a2);
            a3 = _mm256_fmadd_pd(half_vec, a3, b3);
            a3 = _mm256_fmadd_pd(s_vec, c3, a3);
        }

        _mm256_store_pd(a + i, a0);
        _mm256_store_pd(a + i + 4, a1);
        _mm256_store_pd(a + i + 8, a2);
        _mm256_store_pd(a + i + 12, a3);
    }

    // Tail for remaining elements (process 4 at a time)
    for (; i + 3 < N; i += 4) {
        __m256d a_vec = _mm256_load_pd(a + i);
        __m256d b_vec = _mm256_load_pd(b + i);
        __m256d c_vec = _mm256_load_pd(c + i);
        for (int iter = 0; iter < NUM_ITERS; ++iter) {
            a_vec = _mm256_fmadd_pd(half_vec, a_vec, b_vec);
            a_vec = _mm256_fmadd_pd(s_vec, c_vec, a_vec);
        }
        _mm256_store_pd(a + i, a_vec);
    }

    // Scalar tail (should not run for N multiple of 16)
    for (; i < N; ++i) {
        double a_val = a[i];
        const double b_val = b[i];
        const double c_val = c[i];
        for (int iter = 0; iter < NUM_ITERS; ++iter) {
            a_val = 0.5 * a_val + b_val + s * c_val;
        }
        a[i] = a_val;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::printf("Tiempo de ejecucion de la triada (%d iters): %.2f ms\n", NUM_ITERS, elapsed.count());

    double checksum = 0.0;
    for (long j = 0; j < N; ++j) {
        checksum += a[j] * static_cast<double>((j % 13) + 1);
    }
    std::printf("Checksum: %.6f\n", checksum);

    std::free(a);
    std::free(b);
    std::free(c);
    return 0;
}