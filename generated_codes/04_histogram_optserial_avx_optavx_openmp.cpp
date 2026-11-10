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

    srand(42);
    for (long i = 0; i < N; ++i) data[i] = rand() % NUM_BINS;
    for (int b = 0; b < NUM_BINS; ++b) hist[b] = 0;

    double start = omp_get_wtime();

    // SIMD skipped: histogram conflict
    // Swapped loops: outer over data, inner over iterations
    #pragma omp parallel default(none) shared(data, hist)
    {
        int local_hist[NUM_BINS];
        for (int b = 0; b < NUM_BINS; ++b) local_hist[b] = 0;

        #pragma omp for schedule(static)
        for (long i = 0; i < N; ++i) {
            int d = data[i];
            #pragma GCC unroll 50
            for (int iter = 0; iter < NUM_ITERS; ++iter) {
                int bin = (d + iter * 7) & (NUM_BINS - 1);
                local_hist[bin]++;
            }
        }

        #pragma omp critical
        {
            for (int b = 0; b < NUM_BINS; ++b) {
                hist[b] += local_hist[b];
            }
        }
    }

    double end = omp_get_wtime();
    double elapsed_ms = (end - start) * 1000.0;

    printf("Tiempo de ejecucion del histograma (%d iters): %.2f ms\n", NUM_ITERS, elapsed_ms);

    long long checksum = 0;
    for (int b = 0; b < NUM_BINS; ++b) checksum += (long long)hist[b] * (b + 1);
    printf("Checksum: %.6f\n", static_cast<double>(checksum));

    free(data);
    free(hist);
    return 0;
}