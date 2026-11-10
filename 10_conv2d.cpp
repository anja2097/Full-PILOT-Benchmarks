// Convolucion 2D con filtro gaussiano 9x9, aplicada NUM_ITERS veces (desenfoque repetido, doble buffer).
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

int main()
{
    const int N = 4096;
    const int R = 4;                  // radio del filtro: (2R+1) x (2R+1)
    const int W = 2 * R + 1;
    const int NUM_ITERS = 8;

    std::vector<double> filter(W * W);
    double fsum = 0.0;
    for (int dy = -R; dy <= R; ++dy)
        for (int dx = -R; dx <= R; ++dx) {
            double w = std::exp(-(dx * dx + dy * dy) / 8.0);
            filter[(dy + R) * W + (dx + R)] = w;
            fsum += w;
        }
    for (double& w : filter) w /= fsum;

    std::vector<double> in(static_cast<size_t>(N) * N), out(static_cast<size_t>(N) * N);

    srand(42);
    for (size_t p = 0; p < in.size(); ++p) in[p] = static_cast<double>(rand()) / RAND_MAX;
    out = in;   // el borde de anchura R no se modifica

    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < NUM_ITERS; ++iter) {
        for (int i = R; i < N - R; ++i) {
            for (int j = R; j < N - R; ++j) {
                double sum = 0.0;
                for (int dy = -R; dy <= R; ++dy) {
                    for (int dx = -R; dx <= R; ++dx) {
                        sum += filter[(dy + R) * W + (dx + R)] *
                               in[static_cast<size_t>(i + dy) * N + (j + dx)];
                    }
                }
                out[static_cast<size_t>(i) * N + j] = sum;
            }
        }
        std::swap(in, out);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    printf("Tiempo de ejecucion de la convolucion 2D (%d iters): %.2f ms\n", NUM_ITERS, elapsed.count());

    double checksum = 0.0;
    for (size_t p = 0; p < in.size(); ++p) checksum += in[p] * static_cast<double>(p % 11 + 1);
    printf("Checksum: %.6f\n", checksum);
    return 0;
}
