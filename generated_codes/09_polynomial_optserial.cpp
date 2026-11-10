/* ANALYSIS:
 * kernel: degree-31 Horner over 8.4M pts x50 iters (13B FMAs)
 * bottleneck: compute latency (dependent FMA chain) + strided reuse
 * transform: interchange k/i + cache blocking so inner loop is unit-stride FMA
 * pack shifted x into aligned xvbuf; keep r in L1; fuse first/last steps
 * layout: 64B aligned heap via posix_memalign, restrict ptrs, SoA-ready simple loops
 */
#include <chrono>
#include <cstdio>
#include <cstdlib>

#ifndef TILE_L1
#define TILE_L1 1024
#endif

const int DEGREE = 31;
static double coeff[DEGREE + 1];

int main()
{
    const int N = 1 << 23;
    const int NUM_ITERS = 50;

    for (int k = 0; k <= DEGREE; ++k) coeff[k] = 1.0 / (k + 1);

    double *x = nullptr;
    double *acc = nullptr;
    if (posix_memalign((void **)&x, 64, (size_t)N * sizeof(double)) != 0) return 1;
    if (posix_memalign((void **)&acc, 64, (size_t)N * sizeof(double)) != 0) return 1;
    for (int i = 0; i < N; ++i) acc[i] = 0.0;

    srand(42);
    for (int i = 0; i < N; ++i) x[i] = static_cast<double>(rand()) / RAND_MAX;

    auto start = std::chrono::high_resolution_clock::now();

    double * __restrict__ xr = x;
    double * __restrict__ ar = acc;

    alignas(64) double rbuf[TILE_L1];
    alignas(64) double xvbuf[TILE_L1];

    double c_top = coeff[DEGREE];
    double c_next = coeff[DEGREE - 1];

    for (int bi = 0; bi < N; bi += TILE_L1) {
        int B = N - bi;
        if (B > TILE_L1) B = TILE_L1;
        double * __restrict__ xb = xr + bi;
        double * __restrict__ ab = ar + bi;
        for (int iter = 0; iter < NUM_ITERS; ++iter) {
            double shift = 1e-3 * (double)iter;
            for (int j = 0; j < B; ++j) xvbuf[j] = xb[j] + shift;
            double * __restrict__ r = rbuf;
            double * __restrict__ xv = xvbuf;
            for (int j = 0; j < B; ++j) r[j] = c_top * xv[j] + c_next;
            for (int k = DEGREE - 2; k >= 1; --k) {
                double ck = coeff[k];
                for (int j = 0; j < B; ++j) r[j] = r[j] * xv[j] + ck;
            }
            double c0 = coeff[0];
            for (int j = 0; j < B; ++j) {
                r[j] = r[j] * xv[j] + c0;
                ab[j] += r[j];
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    printf("Tiempo de ejecucion del polinomio (%d iters): %.2f ms\n", NUM_ITERS, elapsed.count());

    double checksum = 0.0;
    for (int i = 0; i < N; ++i) checksum += acc[i] * static_cast<double>(i % 7 + 1);
    printf("Checksum: %.6f\n", checksum);
    free(x);
    free(acc);
    return 0;
}