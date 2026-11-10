#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <immintrin.h>
#include <omp.h>

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

    double start = omp_get_wtime();

    const __m256d alpha_v = _mm256_set1_pd(alpha);
    const __m256d beta_v  = _mm256_set1_pd(beta);

    for (int iter = 0; iter < NUM_ITERS; ++iter) {
        double s_prev = s0;  // shared state across tiles for this iteration
        #pragma omp parallel for default(none) schedule(static) ordered \
            shared(x, s_prev) firstprivate(alpha_v, beta_v, alpha_pow_block, s0)
        for (long tile = 0; tile < N / TILE_SIZE; ++tile) {
            #pragma omp ordered
            {
                alignas(64) double temp[K * BLOCK_SIZE];
                long tile_start = tile * TILE_SIZE;
                for (long g = 0; g < (TILE_SIZE / BLOCK_SIZE) / K; ++g) {
                    long base_block = g * K;
                    double* p0 = x + tile_start + (base_block + 0) * BLOCK_SIZE;
                    double* p1 = x + tile_start + (base_block + 1) * BLOCK_SIZE;
                    double* p2 = x + tile_start + (base_block + 2) * BLOCK_SIZE;
                    double* p3 = x + tile_start + (base_block + 3) * BLOCK_SIZE;

                    __m256d t_vec = _mm256_setzero_pd();
                    for (int jj = 0; jj < BLOCK_SIZE; jj += 4) {
                        __m256d r0 = _mm256_load_pd(p0 + jj);
                        __m256d r1 = _mm256_load_pd(p1 + jj);
                        __m256d r2 = _mm256_load_pd(p2 + jj);
                        __m256d r3 = _mm256_load_pd(p3 + jj);
                        __m256d t0 = _mm256_unpacklo_pd(r0, r1);
                        __m256d t1 = _mm256_unpackhi_pd(r0, r1);
                        __m256d t2 = _mm256_unpacklo_pd(r2, r3);
                        __m256d t3 = _mm256_unpackhi_pd(r2, r3);
                        __m256d c0 = _mm256_permute2f128_pd(t0, t2, 0x20);
                        __m256d c1 = _mm256_permute2f128_pd(t1, t3, 0x20);
                        __m256d c2 = _mm256_permute2f128_pd(t0, t2, 0x31);
                        __m256d c3 = _mm256_permute2f128_pd(t1, t3, 0x31);

                        __m256d bc0 = _mm256_mul_pd(beta_v, c0);
                        t_vec = _mm256_fmadd_pd(alpha_v, t_vec, bc0);
                        _mm256_store_pd(&temp[(jj + 0) * K], t_vec);
                        __m256d bc1 = _mm256_mul_pd(beta_v, c1);
                        t_vec = _mm256_fmadd_pd(alpha_v, t_vec, bc1);
                        _mm256_store_pd(&temp[(jj + 1) * K], t_vec);
                        __m256d bc2 = _mm256_mul_pd(beta_v, c2);
                        t_vec = _mm256_fmadd_pd(alpha_v, t_vec, bc2);
                        _mm256_store_pd(&temp[(jj + 2) * K], t_vec);
                        __m256d bc3 = _mm256_mul_pd(beta_v, c3);
                        t_vec = _mm256_fmadd_pd(alpha_v, t_vec, bc3);
                        _mm256_store_pd(&temp[(jj + 3) * K], t_vec);
                    }

                    alignas(32) double t_end_arr[4];
                    _mm256_store_pd(t_end_arr, t_vec);
                    alignas(32) double s_start_arr[K];
                    double s = s_prev;
                    for (int k = 0; k < K; ++k) {
                        s_start_arr[k] = s;
                        double t_end = t_end_arr[k];
                        s = t_end + alpha_pow_block * s;
                    }
                    s_prev = s;

                    __m256d s_start_vec = _mm256_load_pd(s_start_arr);
                    __m256d corr_vec = _mm256_mul_pd(alpha_v, s_start_vec);
                    for (int jj = 0; jj < BLOCK_SIZE; jj += 4) {
                        __m256d v0 = _mm256_load_pd(&temp[(jj + 0) * K]);
                        __m256d v1 = _mm256_load_pd(&temp[(jj + 1) * K]);
                        __m256d v2 = _mm256_load_pd(&temp[(jj + 2) * K]);
                        __m256d v3 = _mm256_load_pd(&temp[(jj + 3) * K]);
                        __m256d corr1 = _mm256_mul_pd(corr_vec, alpha_v);
                        __m256d corr2 = _mm256_mul_pd(corr1, alpha_v);
                        __m256d corr3 = _mm256_mul_pd(corr2, alpha_v);
                        __m256d corr_next = _mm256_mul_pd(corr3, alpha_v);
                        __m256d w0 = _mm256_add_pd(v0, corr_vec);
                        __m256d w1 = _mm256_add_pd(v1, corr1);
                        __m256d w2 = _mm256_add_pd(v2, corr2);
                        __m256d w3 = _mm256_add_pd(v3, corr3);
                        corr_vec = corr_next;
                        __m256d u0 = _mm256_unpacklo_pd(w0, w1);
                        __m256d u1 = _mm256_unpackhi_pd(w0, w1);
                        __m256d u2 = _mm256_unpacklo_pd(w2, w3);
                        __m256d u3 = _mm256_unpackhi_pd(w2, w3);
                        __m256d o0 = _mm256_permute2f128_pd(u0, u2, 0x20);
                        __m256d o1 = _mm256_permute2f128_pd(u1, u3, 0x20);
                        __m256d o2 = _mm256_permute2f128_pd(u0, u2, 0x31);
                        __m256d o3 = _mm256_permute2f128_pd(u1, u3, 0x31);
                        _mm256_store_pd(p0 + jj, o0);
                        _mm256_store_pd(p1 + jj, o1);
                        _mm256_store_pd(p2 + jj, o2);
                        _mm256_store_pd(p3 + jj, o3);
                    }
                }
            }
        }
    }

    double end = omp_get_wtime();
    double elapsed_ms = (end - start) * 1000.0;
    printf("Tiempo de ejecucion del filtro EMA (%d iters): %.2f ms\n", NUM_ITERS, elapsed_ms);

    double checksum = 0.0;
    for (long i = 0; i < N; ++i) checksum += x[i] * static_cast<double>(i % 13 + 1);
    printf("Checksum: %.6f\n", checksum);

    free(x);
    return 0;
}