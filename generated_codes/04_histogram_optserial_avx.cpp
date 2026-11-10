/* ANALYSIS: histogram kernel, memory bandwidth bound, conflict updates on shared bins, SIMD skipped due to histogram conflict */
#include <chrono>
#include <cstdio>
#include <cstdlib>

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

    auto start = std::chrono::high_resolution_clock::now();

    // SIMD skipped: histogram conflict
    // Swapped loops: outer over data, inner over iterations
    for (long i = 0; i < N; ++i) {
        int d = data[i];
        for (int iter = 0; iter < NUM_ITERS; ++iter) {
            int bin = (d + iter * 7) & (NUM_BINS - 1);
            hist[bin]++;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    printf("Tiempo de ejecucion del histograma (%d iters): %.2f ms\n", NUM_ITERS, elapsed.count());

    long long checksum = 0;
    for (int b = 0; b < NUM_BINS; ++b) checksum += (long long)hist[b] * (b + 1);
    printf("Checksum: %.6f\n", static_cast<double>(checksum));

    free(data);
    free(hist);
    return 0;
}