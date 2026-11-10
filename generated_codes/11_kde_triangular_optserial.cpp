// Estimacion de densidad por nucleo gaussiano acumulativa en 2D:
// dens[i] = sum_{j <= i} exp(-|p_i - p_j|^2 / (2 h^2)). Bucle triangular (trabajo desigual por fila).
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main()
{
    const int N = 32768;
    const int NUM_ITERS = 1;
    const double H = 0.05;
    const double INV_2H2 = 1.0 / (2.0 * H * H);

    std::vector<double> px(N), py(N), dens(N, 0.0);

    srand(42);
    for (int i = 0; i < N; ++i) {
        px[i] = static_cast<double>(rand()) / RAND_MAX;
        py[i] = static_cast<double>(rand()) / RAND_MAX;
    }

    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < NUM_ITERS; ++iter) {
        for (int i = 0; i < N; ++i) {
            double sum = 0.0;
            for (int j = 0; j <= i; ++j) {
                double dx = px[i] - px[j];
                double dy = py[i] - py[j];
                sum += std::exp(-(dx * dx + dy * dy) * INV_2H2);
            }
            dens[i] = sum;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    printf("Tiempo de ejecucion de la KDE (%d iters): %.2f ms\n", NUM_ITERS, elapsed.count());

    double checksum = 0.0;
    for (int i = 0; i < N; ++i) checksum += dens[i] * static_cast<double>(i % 5 + 1);
    printf("Checksum: %.6f\n", checksum);
    return 0;
}
