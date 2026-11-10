#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <immintrin.h>
#include <omp.h>

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

    double start = omp_get_wtime();

    double * __restrict__ xr = x;
    double * __restrict__ ar = acc;

    // Hoisted broadcasts of coefficients (constant)
    __m256d vcoeff[DEGREE + 1];
    for (int k = 0; k <= DEGREE; ++k) vcoeff[k] = _mm256_set1_pd(coeff[k]);
    __m256d vctop = vcoeff[DEGREE];
    __m256d vcnext = vcoeff[DEGREE - 1];
    __m256d vc0 = vcoeff[0];

    const int RB = 32; // 8 vectors x 4 doubles

    #pragma omp parallel for schedule(static) default(none) \
        shared(xr, ar, vcoeff, vctop, vcnext, vc0, coeff) \
        firstprivate(N, NUM_ITERS, DEGREE, RB)
    for (int bi = 0; bi < N; bi += TILE_L1) {
        int B = N - bi;
        if (B > TILE_L1) B = TILE_L1;
        double * __restrict__ xb = xr + bi;
        double * __restrict__ ab = ar + bi;
        int jj = 0;
        // main register-blocked part, 32 elements
        for (; jj + RB <= B; jj += RB) {
            __m256d a0 = _mm256_load_pd(ab + jj + 0 * 4);
            __m256d a1 = _mm256_load_pd(ab + jj + 1 * 4);
            __m256d a2 = _mm256_load_pd(ab + jj + 2 * 4);
            __m256d a3 = _mm256_load_pd(ab + jj + 3 * 4);
            __m256d a4 = _mm256_load_pd(ab + jj + 4 * 4);
            __m256d a5 = _mm256_load_pd(ab + jj + 5 * 4);
            __m256d a6 = _mm256_load_pd(ab + jj + 6 * 4);
            __m256d a7 = _mm256_load_pd(ab + jj + 7 * 4);

            for (int iter = 0; iter < NUM_ITERS; ++iter) {
                double shift = 1e-3 * (double)iter;
                __m256d vshift = _mm256_set1_pd(shift);

                __m256d xv0 = _mm256_add_pd(_mm256_load_pd(xb + jj + 0 * 4), vshift);
                __m256d xv1 = _mm256_add_pd(_mm256_load_pd(xb + jj + 1 * 4), vshift);
                __m256d xv2 = _mm256_add_pd(_mm256_load_pd(xb + jj + 2 * 4), vshift);
                __m256d xv3 = _mm256_add_pd(_mm256_load_pd(xb + jj + 3 * 4), vshift);
                __m256d xv4 = _mm256_add_pd(_mm256_load_pd(xb + jj + 4 * 4), vshift);
                __m256d xv5 = _mm256_add_pd(_mm256_load_pd(xb + jj + 5 * 4), vshift);
                __m256d xv6 = _mm256_add_pd(_mm256_load_pd(xb + jj + 6 * 4), vshift);
                __m256d xv7 = _mm256_add_pd(_mm256_load_pd(xb + jj + 7 * 4), vshift);

                __m256d r0 = _mm256_fmadd_pd(vctop, xv0, vcnext);
                __m256d r1 = _mm256_fmadd_pd(vctop, xv1, vcnext);
                __m256d r2 = _mm256_fmadd_pd(vctop, xv2, vcnext);
                __m256d r3 = _mm256_fmadd_pd(vctop, xv3, vcnext);
                __m256d r4 = _mm256_fmadd_pd(vctop, xv4, vcnext);
                __m256d r5 = _mm256_fmadd_pd(vctop, xv5, vcnext);
                __m256d r6 = _mm256_fmadd_pd(vctop, xv6, vcnext);
                __m256d r7 = _mm256_fmadd_pd(vctop, xv7, vcnext);

                for (int k = DEGREE - 2; k >= 1; --k) {
                    __m256d vck = vcoeff[k];
                    r0 = _mm256_fmadd_pd(r0, xv0, vck);
                    r1 = _mm256_fmadd_pd(r1, xv1, vck);
                    r2 = _mm256_fmadd_pd(r2, xv2, vck);
                    r3 = _mm256_fmadd_pd(r3, xv3, vck);
                    r4 = _mm256_fmadd_pd(r4, xv4, vck);
                    r5 = _mm256_fmadd_pd(r5, xv5, vck);
                    r6 = _mm256_fmadd_pd(r6, xv6, vck);
                    r7 = _mm256_fmadd_pd(r7, xv7, vck);
                }

                r0 = _mm256_fmadd_pd(r0, xv0, vc0);
                r1 = _mm256_fmadd_pd(r1, xv1, vc0);
                r2 = _mm256_fmadd_pd(r2, xv2, vc0);
                r3 = _mm256_fmadd_pd(r3, xv3, vc0);
                r4 = _mm256_fmadd_pd(r4, xv4, vc0);
                r5 = _mm256_fmadd_pd(r5, xv5, vc0);
                r6 = _mm256_fmadd_pd(r6, xv6, vc0);
                r7 = _mm256_fmadd_pd(r7, xv7, vc0);

                a0 = _mm256_add_pd(a0, r0);
                a1 = _mm256_add_pd(a1, r1);
                a2 = _mm256_add_pd(a2, r2);
                a3 = _mm256_add_pd(a3, r3);
                a4 = _mm256_add_pd(a4, r4);
                a5 = _mm256_add_pd(a5, r5);
                a6 = _mm256_add_pd(a6, r6);
                a7 = _mm256_add_pd(a7, r7);
            }

            _mm256_store_pd(ab + jj + 0 * 4, a0);
            _mm256_store_pd(ab + jj + 1 * 4, a1);
            _mm256_store_pd(ab + jj + 2 * 4, a2);
            _mm256_store_pd(ab + jj + 3 * 4, a3);
            _mm256_store_pd(ab + jj + 4 * 4, a4);
            _mm256_store_pd(ab + jj + 5 * 4, a5);
            _mm256_store_pd(ab + jj + 6 * 4, a6);
            _mm256_store_pd(ab + jj + 7 * 4, a7);
        }
        // remainder (generic, B is multiple of 32 in practice)
        for (; jj < B; ++jj) {
            double aj = ab[jj];
            double xj = xb[jj];
            for (int iter = 0; iter < NUM_ITERS; ++iter) {
                double shift = 1e-3 * (double)iter;
                double xv = xj + shift;
                double r = coeff[DEGREE] * xv + coeff[DEGREE - 1];
                for (int k = DEGREE - 2; k >= 1; --k) r = r * xv + coeff[k];
                r = r * xv + coeff[0];
                aj += r;
            }
            ab[jj] = aj;
        }
    }

    double end = omp_get_wtime();
    double elapsed_ms = (end - start) * 1000.0;

    printf("Tiempo de ejecucion del polinomio (%d iters): %.2f ms\n", NUM_ITERS, elapsed_ms);

    double checksum = 0.0;
    for (int i = 0; i < N; ++i) checksum += acc[i] * static_cast<double>(i % 7 + 1);
    printf("Checksum: %.6f\n", checksum);
    free(x);
    free(acc);
    return 0;
}