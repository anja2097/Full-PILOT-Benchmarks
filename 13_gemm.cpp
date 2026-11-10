// Producto de matrices denso acumulado: C = A*B + 0.5*C, repetido NUM_ITERS veces.
// Dimensiones distintas y no multiplos de potencias de 2, para que haya bordes en tiles y vectores.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main()
{
    const int M = 1200;           // filas de A y C
    const int N = 1000;           // columnas de B y C
    const int K = 1100;           // columnas de A / filas de B
    const int NUM_ITERS = 5;

    std::vector<double> A(static_cast<size_t>(M) * K);
    std::vector<double> B(static_cast<size_t>(K) * N);
    std::vector<double> C(static_cast<size_t>(M) * N, 0.0);

    srand(42);
    for (size_t p = 0; p < A.size(); ++p) A[p] = static_cast<double>(rand()) / RAND_MAX;
    for (size_t p = 0; p < B.size(); ++p) B[p] = static_cast<double>(rand()) / RAND_MAX;

    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < NUM_ITERS; ++iter) {
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                double sum = 0.0;
                for (int k = 0; k < K; ++k) {
                    sum += A[static_cast<size_t>(i) * K + k] * B[static_cast<size_t>(k) * N + j];
                }
                size_t p = static_cast<size_t>(i) * N + j;
                C[p] = sum + 0.5 * C[p];
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    printf("Tiempo de ejecucion de la GEMM (%d iters): %.2f ms\n", NUM_ITERS, elapsed.count());

    double checksum = 0.0;
    for (size_t p = 0; p < C.size(); ++p) checksum += C[p] * static_cast<double>(p % 13 + 1);
    printf("Checksum: %.6f\n", checksum);
    return 0;
}