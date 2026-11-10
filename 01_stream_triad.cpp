#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main()
{
    const long N = 1L << 25;      // 33.5 M elementos por vector
    const int NUM_ITERS = 100;
    const double s = 0.75;

    std::vector<double> a(N), b(N), c(N);

    srand(42);
    for (long i = 0; i < N; ++i) a[i] = static_cast<double>(rand()) / RAND_MAX;
    for (long i = 0; i < N; ++i) b[i] = static_cast<double>(rand()) / RAND_MAX;
    for (long i = 0; i < N; ++i) c[i] = static_cast<double>(rand()) / RAND_MAX;

    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < NUM_ITERS; ++iter) {
        for (long i = 0; i < N; ++i) {
            a[i] = 0.5 * a[i] + b[i] + s * c[i];
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    printf("Tiempo de ejecucion de la triada (%d iters): %.2f ms\n", NUM_ITERS, elapsed.count());

    double checksum = 0.0;
    for (long i = 0; i < N; ++i) checksum += a[i] * static_cast<double>(i % 13 + 1);
    printf("Checksum: %.6f\n", checksum);
    return 0;
}
