// Histograma de enteros: cada iteracion desplaza las claves y acumula en NUM_BINS contenedores.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main()
{
    const long N = 1L << 26;          // 67 M claves
    const int NUM_BINS = 4096;        // potencia de 2
    const int NUM_ITERS = 50;

    std::vector<int> data(N);
    std::vector<long long> hist(NUM_BINS, 0);

    srand(42);
    for (long i = 0; i < N; ++i) data[i] = rand() % NUM_BINS;

    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < NUM_ITERS; ++iter) {
        for (long i = 0; i < N; ++i) {
            int bin = (data[i] + iter * 7) & (NUM_BINS - 1);
            hist[bin]++;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    printf("Tiempo de ejecucion del histograma (%d iters): %.2f ms\n", NUM_ITERS, elapsed.count());

    long long checksum = 0;
    for (int b = 0; b < NUM_BINS; ++b) checksum += hist[b] * (b + 1);
    printf("Checksum: %.6f\n", static_cast<double>(checksum));
    return 0;
}
