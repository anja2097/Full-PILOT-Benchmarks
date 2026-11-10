/* ANALYSIS: Serial EMA filter, latency-bound due to per‑channel recurrence (prefix‑sum dependence). Vectorized inner‑k loops (size 4) with AVX2 FMA to keep all A‑ops in registers while preserving dependence across j. */
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <immintrin.h>

#define N (1L << 25)
#define NUM_ITERS 40
#define ALPHA 0.999
#define BETA (1.0 - ALPHA)

#define TILE_SIZE 32768
#define BLOCK_SIZE 512
#define K 4

int main()
{
    double* x;
    if (posix_memalign((void**)&x, 64, N * sizeof(double)) != 0) {
        return 1;
    }

    srand(42);
    for (long i = 0; i < N; ++i) x[i] = static_cast<double>(rand()) / RAND_MAX;

    const double alpha = ALPHA;
    const double beta = BETA;
    const double alpha_pow_block = pow(alpha, BLOCK_SIZE);
    const double s0 = x[0];

    auto start = std::chrono::high_resolution_clock::now();

    alignas(64) double temp[K * BLOCK_SIZE];

    const __m256d alpha_v = _mm256_set1_pd(alpha);
    const __m256d beta_v  = _mm256_set1_pd(beta);
    const __m256d one_v   = _mm256_set1_pd(1.0);

    for (int iter = 0; iter < NUM_ITERS; ++iter) {
        double s_prev = s0;
        for (long tile = 0; tile < N / TILE_SIZE; ++tile) {
            long tile_start = tile * TILE_SIZE;
            for (long g = 0; g < (TILE_SIZE / BLOCK_SIZE) / K; ++g) {
                long base_block = g * K;

                // Pack K blocks into interleaved temp (scalar, kept simple)
                for (int k = 0; k < K; ++k) {
                    long block_start = tile_start + (base_block + k) * BLOCK_SIZE;
                    for (int j = 0; j < BLOCK_SIZE; ++j) {
                        temp[j * K + k] = x[block_start + j];
                    }
                }

                // Local scan on K independent chains (vectorized over k)
                __m256d t_vec = _mm256_setzero_pd();            // t[0..K-1]
                for (int j = 0; j < BLOCK_SIZE; ++j) {
                    __m256d tmp_vec = _mm256_loadu_pd(&temp[j * K]);          // load temp[j*K + 0..3]
                    __m256d beta_tmp = _mm256_mul_pd(beta_v, tmp_vec);       // beta * temp
                    t_vec = _mm256_fmadd_pd(alpha_v, t_vec, beta_tmp);       // t = alpha*t + beta*temp
                    _mm256_storeu_pd(&temp[j * K], t_vec);                   // store back
                }

                // Compute block start states and update s_prev (scalar, unchanged)
                double s_start_arr[K];
                double s = s_prev;
                for (int k = 0; k < K; ++k) {
                    s_start_arr[k] = s;
                    double t_end = temp[(BLOCK_SIZE - 1) * K + k];
                    s = t_end + alpha_pow_block * s;
                }
                s_prev = s;

                // Correction (vectorized over k)
                __m256d s_start_vec = _mm256_loadu_pd(s_start_arr);            // s_start_arr[0..3]
                __m256d corr_vec = _mm256_mul_pd(alpha_v, s_start_vec);       // corr = alpha * s_start_arr
                for (int j = 0; j < BLOCK_SIZE; ++j) {
                    __m256d tmp_vec = _mm256_loadu_pd(&temp[j * K]);          // load current values
                    tmp_vec = _mm256_add_pd(tmp_vec, corr_vec);               // add correction
                    _mm256_storeu_pd(&temp[j * K], tmp_vec);                  // store back
                    corr_vec = _mm256_mul_pd(corr_vec, alpha_v);              // corr *= alpha
                }

                // Scatter back to x (scalar, kept simple)
                for (int k = 0; k < K; ++k) {
                    long block_start = tile_start + (base_block + k) * BLOCK_SIZE;
                    for (int j = 0; j < BLOCK_SIZE; ++j) {
                        x[block_start + j] = temp[j * K + k];
                    }
                }
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    printf("Tiempo de ejecucion del filtro EMA (%d iters): %.2f ms\n", NUM_ITERS, elapsed.count());

    double checksum = 0.0;
    for (long i = 0; i < N; ++i) checksum += x[i] * static_cast<double>(i % 13 + 1);
    printf("Checksum: %.6f\n", checksum);

    free(x);
    return 0;
}