// Trasposicion acumulada: B = 0.5*B + A^T, repetida NUM_ITERS veces.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main()
{
    const int N = 6144;
    const int NUM_ITERS = 10;

    std::vector<double> A(static_cast<size_t>(N) * N), B(static_cast<size_t>(N) * N, 0.0);

    srand(42);
    for (size_t p = 0; p < A.size(); ++p) A[p] = static_cast<double>(rand()) / RAND_MAX;

    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < NUM_ITERS; ++iter) {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                size_t dst = static_cast<size_t>(j) * N + i;
                B[dst] = 0.5 * B[dst] + A[static_cast<size_t>(i) * N + j];
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    printf("Tiempo de ejecucion de la trasposicion (%d iters): %.2f ms\n", NUM_ITERS, elapsed.count());

    double checksum = 0.0;
    for (size_t p = 0; p < B.size(); ++p) checksum += B[p] * static_cast<double>(p % 17 + 1);
    printf("Checksum: %.6f\n", checksum);
    return 0;
}
