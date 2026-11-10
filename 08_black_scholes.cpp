// Valoracion Black-Scholes de opciones europeas call y put.
// Cada iteracion desplaza el tipo de interes y acumula los precios.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

// Funcion de distribucion normal acumulada (aproximacion polinomica de Abramowitz-Stegun 26.2.17)
static double cnd(double d)
{
    const double A1 = 0.31938153;
    const double A2 = -0.356563782;
    const double A3 = 1.781477937;
    const double A4 = -1.821255978;
    const double A5 = 1.330274429;
    const double RSQRT2PI = 0.39894228040143267794;

    double k = 1.0 / (1.0 + 0.2316419 * std::fabs(d));
    double poly = k * (A1 + k * (A2 + k * (A3 + k * (A4 + k * A5))));
    double c = RSQRT2PI * std::exp(-0.5 * d * d) * poly;
    return d > 0 ? 1.0 - c : c;
}

int main()
{
    const int N = 1 << 22;            // 4.2 M opciones
    const int NUM_ITERS = 20;

    std::vector<double> S(N), K(N), T(N), R(N), V(N);
    std::vector<double> call_acc(N, 0.0), put_acc(N, 0.0);

    srand(42);
    for (int i = 0; i < N; ++i) {
        S[i] = 10.0 + 100.0 * static_cast<double>(rand()) / RAND_MAX;
        K[i] = 10.0 + 100.0 * static_cast<double>(rand()) / RAND_MAX;
        T[i] = 0.25 + 2.0 * static_cast<double>(rand()) / RAND_MAX;
        R[i] = 0.01 + 0.05 * static_cast<double>(rand()) / RAND_MAX;
        V[i] = 0.10 + 0.40 * static_cast<double>(rand()) / RAND_MAX;
    }

    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < NUM_ITERS; ++iter) {
        for (int i = 0; i < N; ++i) {
            double r = R[i] + 1e-4 * iter;
            double sqrt_t = std::sqrt(T[i]);
            double d1 = (std::log(S[i] / K[i]) + (r + 0.5 * V[i] * V[i]) * T[i]) / (V[i] * sqrt_t);
            double d2 = d1 - V[i] * sqrt_t;
            double disc = K[i] * std::exp(-r * T[i]);
            double call = S[i] * cnd(d1) - disc * cnd(d2);
            double put = disc * cnd(-d2) - S[i] * cnd(-d1);
            call_acc[i] += call;
            put_acc[i] += put;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    printf("Tiempo de ejecucion de Black-Scholes (%d iters): %.2f ms\n", NUM_ITERS, elapsed.count());

    double checksum = 0.0;
    for (int i = 0; i < N; ++i) checksum += call_acc[i] + put_acc[i];
    printf("Checksum: %.6f\n", checksum);
    return 0;
}
