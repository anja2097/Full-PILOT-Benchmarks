// Producto matriz dispersa (CSR) por vector iterado: x <- A*x + b.
// Columnas aleatorias (acceso irregular a x); cada fila suma 0.9, por lo que la iteracion es estable.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

int main()
{
    const int N = 1 << 21;            // 2.1 M filas
    const int NNZ_PER_ROW = 16;
    const int NUM_ITERS = 30;
    const long NNZ = static_cast<long>(N) * NNZ_PER_ROW;

    std::vector<long> row_ptr(N + 1);
    std::vector<int> col(NNZ);
    std::vector<double> val(NNZ);
    std::vector<double> x(N), y(N), b(N);

    srand(42);
    for (int i = 0; i < N; ++i) {
        row_ptr[i] = static_cast<long>(i) * NNZ_PER_ROW;
        double row_sum = 0.0;
        for (int k = 0; k < NNZ_PER_ROW; ++k) {
            long idx = row_ptr[i] + k;
            col[idx] = rand() % N;
            val[idx] = 0.1 + static_cast<double>(rand()) / RAND_MAX;
            row_sum += val[idx];
        }
        for (int k = 0; k < NNZ_PER_ROW; ++k) val[row_ptr[i] + k] *= 0.9 / row_sum;
    }
    row_ptr[N] = NNZ;
    for (int i = 0; i < N; ++i) x[i] = static_cast<double>(rand()) / RAND_MAX;
    for (int i = 0; i < N; ++i) b[i] = static_cast<double>(rand()) / RAND_MAX;

    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < NUM_ITERS; ++iter) {
        for (int i = 0; i < N; ++i) {
            double sum = 0.0;
            for (long k = row_ptr[i]; k < row_ptr[i + 1]; ++k) {
                sum += val[k] * x[col[k]];
            }
            y[i] = sum + b[i];
        }
        std::swap(x, y);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    printf("Tiempo de ejecucion de la SpMV (%d iters): %.2f ms\n", NUM_ITERS, elapsed.count());

    double checksum = 0.0;
    for (int i = 0; i < N; ++i) checksum += x[i] * static_cast<double>(i % 11 + 1);
    printf("Checksum: %.6f\n", checksum);
    return 0;
}
