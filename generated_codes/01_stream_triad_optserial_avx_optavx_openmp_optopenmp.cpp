/* ANALYSIS:
 * Kernel: STREAM triad-like recurrence, 100 iterations of a = 0.5*a + b + s*c.
 * Bottleneck: compute-bound, FMA latency. Original 4 accumulators give only 1 FMA/cycle.
 * Transforms: increased register blocking to 8 accumulators (32 elements/iter) to
 * reach 2 FMA/cycle. Added parallel first-touch for NUMA locality. Static schedule.
 */
#include <cstdio>
#include <cstdlib>
#include <immintrin.h>
#include <omp.h>

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

    // Parallel first-touch to place pages on the NUMA node of the thread that will use them.
    long vec_iters = N / 32;  // 32 elements per iteration (8 vectors of 4 doubles)
    #pragma omp parallel for schedule(static)
    for (long idx = 0; idx < vec_iters; ++idx) {
        long i = idx * 32;
        for (int k = 0; k < 32; ++k) {
            a[i+k] = 0.0;
            b[i+k] = 0.0;
            c[i+k] = 0.0;
        }
    }

    std::srand(42);
    for (long i = 0; i < N; ++i) a[i] = static_cast<double>(std::rand()) / RAND_MAX;
    for (long i = 0; i < N; ++i) b[i] = static_cast<double>(std::rand()) / RAND_MAX;
    for (long i = 0; i < N; ++i) c[i] = static_cast<double>(std::rand()) / RAND_MAX;

    double start = omp_get_wtime();

    const __m256d half_vec = _mm256_set1_pd(0.5);
    const __m256d s_vec    = _mm256_set1_pd(s);

    #pragma omp parallel for schedule(static) default(none) shared(vec_iters,a,b,c) firstprivate(half_vec,s_vec)
    for (long idx = 0; idx < vec_iters; ++idx) {
        long i = idx * 32;

        __m256d a0 = _mm256_load_pd(a + i);
        __m256d a1 = _mm256_load_pd(a + i + 4);
        __m256d a2 = _mm256_load_pd(a + i + 8);
        __m256d a3 = _mm256_load_pd(a + i + 12);
        __m256d a4 = _mm256_load_pd(a + i + 16);
        __m256d a5 = _mm256_load_pd(a + i + 20);
        __m256d a6 = _mm256_load_pd(a + i + 24);
        __m256d a7 = _mm256_load_pd(a + i + 28);
        __m256d b0 = _mm256_load_pd(b + i);
        __m256d b1 = _mm256_load_pd(b + i + 4);
        __m256d b2 = _mm256_load_pd(b + i + 8);
        __m256d b3 = _mm256_load_pd(b + i + 12);
        __m256d b4 = _mm256_load_pd(b + i + 16);
        __m256d b5 = _mm256_load_pd(b + i + 20);
        __m256d b6 = _mm256_load_pd(b + i + 24);
        __m256d b7 = _mm256_load_pd(b + i + 28);
        __m256d c0 = _mm256_load_pd(c + i);
        __m256d c1 = _mm256_load_pd(c + i + 4);
        __m256d c2 = _mm256_load_pd(c + i + 8);
        __m256d c3 = _mm256_load_pd(c + i + 12);
        __m256d c4 = _mm256_load_pd(c + i + 16);
        __m256d c5 = _mm256_load_pd(c + i + 20);
        __m256d c6 = _mm256_load_pd(c + i + 24);
        __m256d c7 = _mm256_load_pd(c + i + 28);

        for (int iter = 0; iter < NUM_ITERS; ++iter) {
            a0 = _mm256_fmadd_pd(half_vec, a0, b0);
            a0 = _mm256_fmadd_pd(s_vec, c0, a0);
            a1 = _mm256_fmadd_pd(half_vec, a1, b1);
            a1 = _mm256_fmadd_pd(s_vec, c1, a1);
            a2 = _mm256_fmadd_pd(half_vec, a2, b2);
            a2 = _mm256_fmadd_pd(s_vec, c2, a2);
            a3 = _mm256_fmadd_pd(half_vec, a3, b3);
            a3 = _mm256_fmadd_pd(s_vec, c3, a3);
            a4 = _mm256_fmadd_pd(half_vec, a4, b4);
            a4 = _mm256_fmadd_pd(s_vec, c4, a4);
            a5 = _mm256_fmadd_pd(half_vec, a5, b5);
            a5 = _mm256_fmadd_pd(s_vec, c5, a5);
            a6 = _mm256_fmadd_pd(half_vec, a6, b6);
            a6 = _mm256_fmadd_pd(s_vec, c6, a6);
            a7 = _mm256_fmadd_pd(half_vec, a7, b7);
            a7 = _mm256_fmadd_pd(s_vec, c7, a7);
        }

        _mm256_store_pd(a + i, a0);
        _mm256_store_pd(a + i + 4, a1);
        _mm256_store_pd(a + i + 8, a2);
        _mm256_store_pd(a + i + 12, a3);
        _mm256_store_pd(a + i + 16, a4);
        _mm256_store_pd(a + i + 20, a5);
        _mm256_store_pd(a + i + 24, a6);
        _mm256_store_pd(a + i + 28, a7);
    }

    // Tail for remaining elements (N is a multiple of 32, so this will not run)
    long i = vec_iters * 32;
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
    for (; i < N; ++i) {
        double a_val = a[i];
        const double b_val = b[i];
        const double c_val = c[i];
        for (int iter = 0; iter < NUM_ITERS; ++iter) {
            a_val = 0.5 * a_val + b_val + s * c_val;
        }
        a[i] = a_val;
    }

    double end = omp_get_wtime();
    double elapsed_ms = (end - start) * 1000.0;
    std::printf("Tiempo de ejecucion de la triada (%d iters): %.2f ms\n", NUM_ITERS, elapsed_ms);

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