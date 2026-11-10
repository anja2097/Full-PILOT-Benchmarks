// Media movil exponencial (filtro IIR de primer orden) aplicada en el sitio, repetida NUM_ITERS veces.
// Cada elemento depende del anterior: recurrencia lineal s = ALPHA*s + (1-ALPHA)*x[i].
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main()
{
    const long N = 1L << 25;          // 33.5 M muestras
    const int NUM_ITERS = 40;
    const double ALPHA = 0.999;

    std::vector<double> x(N);

    srand(42);
    for (long i = 0; i < N; ++i) x[i] = static_cast<double>(rand()) / RAND_MAX;

    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < NUM_ITERS; ++iter) {
        double s = x[0];
        for (long i = 0; i < N; ++i) {
            s = ALPHA * s + (1.0 - ALPHA) * x[i];
            x[i] = s;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    printf("Tiempo de ejecucion del filtro EMA (%d iters): %.2f ms\n", NUM_ITERS, elapsed.count());

    double checksum = 0.0;
    for (long i = 0; i < N; ++i) checksum += x[i] * static_cast<double>(i % 13 + 1);
    printf("Checksum: %.6f\n", checksum);
    return 0;
}
