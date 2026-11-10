#include <immintrin.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <omp.h>

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

    double start = omp_get_wtime();

    const __m256d neg_inv = _mm256_set1_pd(-INV_2H2);

    for (int iter = 0; iter < NUM_ITERS; ++iter) {
        #pragma omp parallel for schedule(static) default(none) shared(px, py, dens)
        for (int i = 0; i < N; ++i) {
            double xi = px[i];
            double yi = py[i];
            __m256d vx = _mm256_set1_pd(xi);
            __m256d vy = _mm256_set1_pd(yi);

            __m256d sum0 = _mm256_setzero_pd();
            __m256d sum1 = _mm256_setzero_pd();
            __m256d sum2 = _mm256_setzero_pd();
            __m256d sum3 = _mm256_setzero_pd();

            int limit = i + 1;
            int j = 0;

            // 16-wide main loop
            for (; j + 16 <= limit; j += 16) {
                __m256d vpx0 = _mm256_loadu_pd(px.data() + j);
                __m256d vpy0 = _mm256_loadu_pd(py.data() + j);
                __m256d vpx1 = _mm256_loadu_pd(px.data() + j + 4);
                __m256d vpy1 = _mm256_loadu_pd(py.data() + j + 4);
                __m256d vpx2 = _mm256_loadu_pd(px.data() + j + 8);
                __m256d vpy2 = _mm256_loadu_pd(py.data() + j + 8);
                __m256d vpx3 = _mm256_loadu_pd(px.data() + j + 12);
                __m256d vpy3 = _mm256_loadu_pd(py.data() + j + 12);

                __m256d dx0 = _mm256_sub_pd(vx, vpx0);
                __m256d dy0 = _mm256_sub_pd(vy, vpy0);
                __m256d d20 = _mm256_fmadd_pd(dx0, dx0, _mm256_mul_pd(dy0, dy0));
                __m256d arg0 = _mm256_mul_pd(d20, neg_inv);

                __m256d dx1 = _mm256_sub_pd(vx, vpx1);
                __m256d dy1 = _mm256_sub_pd(vy, vpy1);
                __m256d d21 = _mm256_fmadd_pd(dx1, dx1, _mm256_mul_pd(dy1, dy1));
                __m256d arg1 = _mm256_mul_pd(d21, neg_inv);

                __m256d dx2 = _mm256_sub_pd(vx, vpx2);
                __m256d dy2 = _mm256_sub_pd(vy, vpy2);
                __m256d d22 = _mm256_fmadd_pd(dx2, dx2, _mm256_mul_pd(dy2, dy2));
                __m256d arg2 = _mm256_mul_pd(d22, neg_inv);

                __m256d dx3 = _mm256_sub_pd(vx, vpx3);
                __m256d dy3 = _mm256_sub_pd(vy, vpy3);
                __m256d d23 = _mm256_fmadd_pd(dx3, dx3, _mm256_mul_pd(dy3, dy3));
                __m256d arg3 = _mm256_mul_pd(d23, neg_inv);

                alignas(32) double tmp[16];
                _mm256_store_pd(tmp, arg0);
                _mm256_store_pd(tmp + 4, arg1);
                _mm256_store_pd(tmp + 8, arg2);
                _mm256_store_pd(tmp + 12, arg3);

                tmp[0]  = std::exp(tmp[0]);
                tmp[1]  = std::exp(tmp[1]);
                tmp[2]  = std::exp(tmp[2]);
                tmp[3]  = std::exp(tmp[3]);
                tmp[4]  = std::exp(tmp[4]);
                tmp[5]  = std::exp(tmp[5]);
                tmp[6]  = std::exp(tmp[6]);
                tmp[7]  = std::exp(tmp[7]);
                tmp[8]  = std::exp(tmp[8]);
                tmp[9]  = std::exp(tmp[9]);
                tmp[10] = std::exp(tmp[10]);
                tmp[11] = std::exp(tmp[11]);
                tmp[12] = std::exp(tmp[12]);
                tmp[13] = std::exp(tmp[13]);
                tmp[14] = std::exp(tmp[14]);
                tmp[15] = std::exp(tmp[15]);

                __m256d exp0 = _mm256_load_pd(tmp);
                __m256d exp1 = _mm256_load_pd(tmp + 4);
                __m256d exp2 = _mm256_load_pd(tmp + 8);
                __m256d exp3 = _mm256_load_pd(tmp + 12);

                sum0 = _mm256_add_pd(sum0, exp0);
                sum1 = _mm256_add_pd(sum1, exp1);
                sum2 = _mm256_add_pd(sum2, exp2);
                sum3 = _mm256_add_pd(sum3, exp3);
            }

            // 8-wide loop
            for (; j + 8 <= limit; j += 8) {
                __m256d vpx0 = _mm256_loadu_pd(px.data() + j);
                __m256d vpy0 = _mm256_loadu_pd(py.data() + j);
                __m256d vpx1 = _mm256_loadu_pd(px.data() + j + 4);
                __m256d vpy1 = _mm256_loadu_pd(py.data() + j + 4);

                __m256d dx0 = _mm256_sub_pd(vx, vpx0);
                __m256d dy0 = _mm256_sub_pd(vy, vpy0);
                __m256d d20 = _mm256_fmadd_pd(dx0, dx0, _mm256_mul_pd(dy0, dy0));
                __m256d arg0 = _mm256_mul_pd(d20, neg_inv);

                __m256d dx1 = _mm256_sub_pd(vx, vpx1);
                __m256d dy1 = _mm256_sub_pd(vy, vpy1);
                __m256d d21 = _mm256_fmadd_pd(dx1, dx1, _mm256_mul_pd(dy1, dy1));
                __m256d arg1 = _mm256_mul_pd(d21, neg_inv);

                alignas(32) double tmp[8];
                _mm256_store_pd(tmp, arg0);
                _mm256_store_pd(tmp + 4, arg1);

                tmp[0] = std::exp(tmp[0]);
                tmp[1] = std::exp(tmp[1]);
                tmp[2] = std::exp(tmp[2]);
                tmp[3] = std::exp(tmp[3]);
                tmp[4] = std::exp(tmp[4]);
                tmp[5] = std::exp(tmp[5]);
                tmp[6] = std::exp(tmp[6]);
                tmp[7] = std::exp(tmp[7]);

                __m256d exp0 = _mm256_load_pd(tmp);
                __m256d exp1 = _mm256_load_pd(tmp + 4);

                sum0 = _mm256_add_pd(sum0, exp0);
                sum1 = _mm256_add_pd(sum1, exp1);
            }

            // 4-wide loop
            for (; j + 4 <= limit; j += 4) {
                __m256d vpx = _mm256_loadu_pd(px.data() + j);
                __m256d vpy = _mm256_loadu_pd(py.data() + j);

                __m256d dx = _mm256_sub_pd(vx, vpx);
                __m256d dy = _mm256_sub_pd(vy, vpy);
                __m256d d2 = _mm256_fmadd_pd(dx, dx, _mm256_mul_pd(dy, dy));
                __m256d arg = _mm256_mul_pd(d2, neg_inv);

                alignas(32) double tmp[4];
                _mm256_store_pd(tmp, arg);

                tmp[0] = std::exp(tmp[0]);
                tmp[1] = std::exp(tmp[1]);
                tmp[2] = std::exp(tmp[2]);
                tmp[3] = std::exp(tmp[3]);

                __m256d expv = _mm256_load_pd(tmp);
                sum0 = _mm256_add_pd(sum0, expv);
            }

            // Combine vector accumulators
            __m256d sum_vec = _mm256_add_pd(_mm256_add_pd(sum0, sum1),
                                            _mm256_add_pd(sum2, sum3));
            alignas(32) double sum_arr[4];
            _mm256_storeu_pd(sum_arr, sum_vec);
            double sum = sum_arr[0] + sum_arr[1] + sum_arr[2] + sum_arr[3];

            // Scalar tail
            for (; j < limit; ++j) {
                double dx = xi - px[j];
                double dy = yi - py[j];
                sum += std::exp(-(dx * dx + dy * dy) * INV_2H2);
            }

            dens[i] = sum;
        }
    }

    double end = omp_get_wtime();
    double elapsed_ms = (end - start) * 1000.0;

    printf("Tiempo de ejecucion de la KDE (%d iters): %.2f ms\n", NUM_ITERS, elapsed_ms);

    double checksum = 0.0;
    for (int i = 0; i < N; ++i) checksum += dens[i] * static_cast<double>(i % 5 + 1);
    printf("Checksum: %.6f\n", checksum);
    return 0;
}