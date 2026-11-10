// Evaluacion de un polinomio de grado 31 (esquema de Horner) sobre un vector, acumulando resultados.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

const int DEGREE = 31;
static double coeff[DEGREE + 1];

static double poly_eval(double x)
{
    double r = coeff[DEGREE];
    for (int k = DEGREE - 1; k >= 0; --k) {
        r = r * x + coeff[k];
    }
    return r;
}

int main()
{
    const int N = 1 << 23;            // 8.4 M puntos
    const int NUM_ITERS = 50;

    for (int k = 0; k <= DEGREE; ++k) coeff[k] = 1.0 / (k + 1);

    std::vector<double> x(N), acc(N, 0.0);

    srand(42);
    for (int i = 0; i < N; ++i) x[i] = static_cast<double>(rand()) / RAND_MAX;

    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < NUM_ITERS; ++iter) {
        double shift = 1e-3 * iter;
        for (int i = 0; i < N; ++i) {
            acc[i] += poly_eval(x[i] + shift);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    printf("Tiempo de ejecucion del polinomio (%d iters): %.2f ms\n", NUM_ITERS, elapsed.count());

    double checksum = 0.0;
    for (int i = 0; i < N; ++i) checksum += acc[i] * static_cast<double>(i % 7 + 1);
    printf("Checksum: %.6f\n", checksum);
    return 0;
}
