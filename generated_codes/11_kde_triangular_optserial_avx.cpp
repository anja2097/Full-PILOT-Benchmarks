#include <immintrin.h>
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
            double xi = px[i];
            double yi = py[i];
            __m256d vx = _mm256_set1_pd(xi);
            __m256d vy = _mm256_set1_pd(yi);
            __m256d sum_vec = _mm256_setzero_pd();

            int limit = i + 1;
            int vec_end = limit & ~3;               // largest multiple of 4 <= limit

            for (int j = 0; j < vec_end; j += 4) {
                __m256d vpx = _mm256_loadu_pd(px.data() + j);
                __m256d vpy = _mm256_loadu_pd(py.data() + j);
                __m256d dx = _mm256_sub_pd(vx, vpx);
                __m256d dy = _mm256_sub_pd(vy, vpy);
                __m256d d2 = _mm256_fmadd_pd(dx, dx, _mm256_mul_pd(dy, dy)); // dx*dx + dy*dy
                __m256d arg = _mm256_mul_pd(d2, _mm256_set1_pd(-INV_2H2));

                alignas(32) double tmp[4];
                _mm256_storeu_pd(tmp, arg);
                for (int k = 0; k < 4; ++k) {
                    tmp[k] = std::exp(tmp[k]);       // class‑C scalar libm
                }
                __m256d expv = _mm256_loadu_pd(tmp);
                sum_vec = _mm256_add_pd(sum_vec, expv);
            }

            double sum = 0.0;
            for (int j = vec_end; j < limit; ++j) {
                double dx = xi - px[j];
                double dy = yi - py[j];
                sum += std::exp(-(dx * dx + dy * dy) * INV_2H2);
            }

            alignas(32) double sum_arr[4];
            _mm256_storeu_pd(sum_arr, sum_vec);
            for (int k = 0; k < 4; ++k) sum += sum_arr[k];

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