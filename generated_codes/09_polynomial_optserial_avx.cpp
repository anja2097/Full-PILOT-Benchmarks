/* ANALYSIS: AVX2 vectorization of polynomial evaluation (compute bound). 
   Transform scalar per-element Horner loops into SIMD across 4 lanes.
   Use aligned loads/stores, broadcast constants, and FMA intrinsics.
   Tail handled with scalar loop. */

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <immintrin.h>

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

    const int VEC = 4; // AVX2 width for doubles

    for (int bi = 0; bi < N; bi += TILE_L1) {
        int B = N - bi;
        if (B > TILE_L1) B = TILE_L1;
        double * __restrict__ xb = xr + bi;
        double * __restrict__ ab = ar + bi;
        for (int iter = 0; iter < NUM_ITERS; ++iter) {
            double shift = 1e-3 * (double)iter;

            // xvbuf = xb + shift
            __m256d vshift = _mm256_set1_pd(shift);
            int j = 0;
            for (; j + VEC <= B; j += VEC) {
                __m256d vx = _mm256_load_pd(xb + j);
                __m256d vxp = _mm256_add_pd(vx, vshift);
                _mm256_store_pd(xvbuf + j, vxp);
            }
            for (; j < B; ++j) {
                xvbuf[j] = xb[j] + shift;
            }

            // r = c_top * xv + c_next
            __m256d vctop  = _mm256_set1_pd(c_top);
            __m256d vcnext = _mm256_set1_pd(c_next);
            j = 0;
            for (; j + VEC <= B; j += VEC) {
                __m256d vx = _mm256_load_pd(xvbuf + j);
                __m256d vr = _mm256_fmadd_pd(vctop, vx, vcnext);
                _mm256_store_pd(rbuf + j, vr);
            }
            for (; j < B; ++j) {
                rbuf[j] = c_top * xvbuf[j] + c_next;
            }

            // Horner loop for k = DEGREE-2 .. 1
            for (int k = DEGREE - 2; k >= 1; --k) {
                __m256d vck = _mm256_set1_pd(coeff[k]);
                j = 0;
                for (; j + VEC <= B; j += VEC) {
                    __m256d vr = _mm256_load_pd(rbuf + j);
                    __m256d vx = _mm256_load_pd(xvbuf + j);
                    vr = _mm256_fmadd_pd(vr, vx, vck);
                    _mm256_store_pd(rbuf + j, vr);
                }
                for (; j < B; ++j) {
                    rbuf[j] = rbuf[j] * xvbuf[j] + coeff[k];
                }
            }

            // Final coefficient and accumulation into acc
            __m256d vc0 = _mm256_set1_pd(coeff[0]);
            j = 0;
            for (; j + VEC <= B; j += VEC) {
                __m256d vr = _mm256_load_pd(rbuf + j);
                __m256d vx = _mm256_load_pd(xvbuf + j);
                vr = _mm256_fmadd_pd(vr, vx, vc0);
                __m256d vab = _mm256_load_pd(ab + j);
                vab = _mm256_add_pd(vab, vr);
                _mm256_store_pd(ab + j, vab);
            }
            for (; j < B; ++j) {
                rbuf[j] = rbuf[j] * xvbuf[j] + coeff[0];
                ab[j] += rbuf[j];
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