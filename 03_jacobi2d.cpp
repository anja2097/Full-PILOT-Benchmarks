// Stencil de Jacobi 2D de 5 puntos con doble buffer y borde fijo.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

int main()
{
    const int N = 4096;           // malla N x N incluyendo el borde
    const int NUM_ITERS = 200;

    std::vector<double> A(static_cast<size_t>(N) * N), B(static_cast<size_t>(N) * N);

    srand(42);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            bool border = (i == 0 || j == 0 || i == N - 1 || j == N - 1);
            A[static_cast<size_t>(i) * N + j] = border ? 1.0 : static_cast<double>(rand()) / RAND_MAX;
        }
    }
    B = A;

    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < NUM_ITERS; ++iter) {
        for (int i = 1; i < N - 1; ++i) {
            for (int j = 1; j < N - 1; ++j) {
                size_t p = static_cast<size_t>(i) * N + j;
                B[p] = 0.2 * (A[p] + A[p - 1] + A[p + 1] + A[p - N] + A[p + N]);
            }
        }
        std::swap(A, B);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    printf("Tiempo de ejecucion de Jacobi 2D (%d iters): %.2f ms\n", NUM_ITERS, elapsed.count());

    double checksum = 0.0;
    for (size_t p = 0; p < A.size(); ++p) checksum += A[p] * static_cast<double>(p % 7 + 1);
    printf("Checksum: %.6f\n", checksum);
    return 0;
}
