// Floyd-Warshall: caminos minimos entre todos los pares de un grafo denso con pesos enteros.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main()
{
    const int N = 3072;

    std::vector<int> dist(static_cast<size_t>(N) * N);

    srand(42);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            dist[static_cast<size_t>(i) * N + j] = (i == j) ? 0 : 1 + rand() % 1000;

    auto start = std::chrono::high_resolution_clock::now();

    for (int k = 0; k < N; ++k) {
        for (int i = 0; i < N; ++i) {
            int dik = dist[static_cast<size_t>(i) * N + k];
            for (int j = 0; j < N; ++j) {
                int via = dik + dist[static_cast<size_t>(k) * N + j];
                int& dij = dist[static_cast<size_t>(i) * N + j];
                dij = std::min(dij, via);
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    printf("Tiempo de ejecucion de Floyd-Warshall (N=%d): %.2f ms\n", N, elapsed.count());

    long long checksum = 0;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            checksum += static_cast<long long>(dist[static_cast<size_t>(i) * N + j]) * ((i + j) % 7 + 1);
    printf("Checksum: %.6f\n", static_cast<double>(checksum));
    return 0;
}
