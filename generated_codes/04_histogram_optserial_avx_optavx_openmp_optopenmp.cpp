/* ANALYSIS: Histogram computation with loop-carried dependencies preventing SIMD. 
 * Bottleneck: Critical section for histogram reduction causing serialization. 
 * Applied: 
 *   - Replaced critical section with OpenMP array reduction to enable parallel updates.
 *   - Added NUMA-aware first-touch placement for data and hist arrays to improve memory locality.
 *   - Kept loop unrolling and schedule(static) for uniform work distribution.
 */

#include <cstdio>
#include <cstdlib>
#include <omp.h>

int main()
{
    const long N = 1L << 26;
    const int NUM_BINS = 4096;
    const int NUM_ITERS = 50;

    int* __restrict__ data;
    int* __restrict__ hist;

    if (posix_memalign((void**)&data, 64, N * sizeof(int)) != 0) return 1;
    if (posix_memalign((void**)&hist, 64, NUM_BINS * sizeof(int)) != 0) return 1;

    // First-touch for data: place pages in parallel with same mapping as timed kernel
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < N; ++i) {
        data[i] = 0;
    }

    srand(42);
    for (long i = 0; i < N; ++i) {
        data[i] = rand() % NUM_BINS;
    }

    // First-touch for hist: place pages in parallel with same mapping as timed kernel
    #pragma omp parallel for schedule(static)
    for (int b = 0; b < NUM_BINS; ++b) {
        hist[b] = 0;
    }

    double start = omp_get_wtime();

    #pragma omp parallel for reduction(+:hist[0:NUM_BINS]) schedule(static)
    for (long i = 0; i < N; ++i) {
        int d = data[i];
        #pragma GCC unroll 50
        for (int iter = 0; iter < NUM_ITERS; ++iter) {
            int bin = (d + iter * 7) & (NUM_BINS - 1);
            hist[bin]++;
        }
    }

    double end = omp_get_wtime();
    double elapsed_ms = (end - start) * 1000.0;

    printf("Tiempo de ejecucion del histograma (%d iters): %.2f ms\n", NUM_ITERS, elapsed_ms);

    long long checksum = 0;
    for (int b = 0; b < NUM_BINS; ++b) {
        checksum += (long long)hist[b] * (b + 1);
    }
    printf("Checksum: %.6f\n", static_cast<double>(checksum));

    free(data);
    free(hist);
    return 0;
}