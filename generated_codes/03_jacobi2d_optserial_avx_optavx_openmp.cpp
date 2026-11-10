#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <immintrin.h>
#include <omp.h>

#ifndef BLOCK_ROWS
#define BLOCK_ROWS 48
#endif
#ifndef TIME_TILE
#define TIME_TILE 8
#endif

int main()
{
    const int N = 4096;
    const int NUM_ITERS = 200;

    double *A_raw = nullptr, *B_raw = nullptr;
    posix_memalign((void**)&A_raw, 64, (size_t)N * N * sizeof(double));
    posix_memalign((void**)&B_raw, 64, (size_t)N * N * sizeof(double));
    double * __restrict__ A = A_raw;
    double * __restrict__ B = B_raw;

    srand(42);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            bool border = (i == 0 || j == 0 || i == N - 1 || j == N - 1);
            A[(size_t)i * N + j] = border ? 1.0 : (double)rand() / RAND_MAX;
        }
    }
    memcpy(B, A, (size_t)N * N * sizeof(double));

    const int Hmax = BLOCK_ROWS + 2 * TIME_TILE;
    double *t0_raw = nullptr, *t1_raw = nullptr;
    posix_memalign((void**)&t0_raw, 64, (size_t)Hmax * N * sizeof(double));
    posix_memalign((void**)&t1_raw, 64, (size_t)Hmax * N * sizeof(double));
    double * __restrict__ tmp0 = t0_raw;
    double * __restrict__ tmp1 = t1_raw;

    double * __restrict__ curSrc = A;
    double * __restrict__ curDst = B;

    double start = omp_get_wtime();

    for (int t0 = 0; t0 < NUM_ITERS; t0 += TIME_TILE) {
        int Tcur = std::min(TIME_TILE, NUM_ITERS - t0);
        for (int b0 = 1; b0 < N - 1; b0 += BLOCK_ROWS) {
            int b1 = std::min(b0 + BLOCK_ROWS, N - 1);
            int e0 = b0 - Tcur;
            if (e0 < 0) e0 = 0;
            int e1 = b1 + Tcur;
            if (e1 > N) e1 = N;
            int H = e1 - e0;
            memcpy(tmp0, curSrc + (size_t)e0 * N, (size_t)H * N * sizeof(double));
            memcpy(tmp1, curSrc + (size_t)e0 * N, (size_t)H * N * sizeof(double));

            double * __restrict__ sBuf = tmp0;
            double * __restrict__ dBuf = tmp1;
            const __m256d coeff = _mm256_set1_pd(0.2);
            for (int s = 0; s < Tcur; ++s) {
                int c0 = b0 - (Tcur - 1 - s);
                if (c0 < 1) c0 = 1;
                int c1 = b1 + (Tcur - 1 - s);
                if (c1 > N - 1) c1 = N - 1;

                #pragma omp parallel for schedule(static) default(none) \
                    shared(c0,c1,dBuf,sBuf,e0)
                for (int i = c0; i < c1; ++i) {
                    size_t li = (size_t)(i - e0);
                    double * __restrict__ dRow = dBuf + li * N;
                    const double * __restrict__ sRow = sBuf + li * N;
                    const double * __restrict__ sRowM = sRow - N;
                    const double * __restrict__ sRowP = sRow + N;
                    dRow[0] = sRow[0];
                    dRow[N - 1] = sRow[N - 1];

                    int j = 1;
                    int vec_end = N - 5; // last j where j+3 <= N-2
                    for (; j <= vec_end; j += 4) {
                        __m256d s_center = _mm256_loadu_pd(&sRow[j]);
                        __m256d s_left   = _mm256_loadu_pd(&sRow[j - 1]);
                        __m256d s_right  = _mm256_loadu_pd(&sRow[j + 1]);
                        __m256d s_up     = _mm256_loadu_pd(&sRowM[j]);
                        __m256d s_down   = _mm256_loadu_pd(&sRowP[j]);

                        __m256d sum = _mm256_add_pd(s_center, s_left);
                        sum = _mm256_add_pd(sum, s_right);
                        sum = _mm256_add_pd(sum, s_up);
                        sum = _mm256_add_pd(sum, s_down);
                        __m256d res = _mm256_mul_pd(coeff, sum);
                        _mm256_storeu_pd(&dRow[j], res);
                    }
                    for (; j < N - 1; ++j) {
                        dRow[j] = 0.2 *
                                  (sRow[j] + sRow[j - 1] + sRow[j + 1] +
                                   sRowM[j] + sRowP[j]);
                    }
                }

                for (int i = e0; i < c0; ++i) {
                    size_t li = (size_t)(i - e0);
                    memcpy(dBuf + li * N, sBuf + li * N, (size_t)N * sizeof(double));
                }
                for (int i = c1; i < e1; ++i) {
                    size_t li = (size_t)(i - e0);
                    memcpy(dBuf + li * N, sBuf + li * N, (size_t)N * sizeof(double));
                }
                double * __restrict__ tmpS = sBuf;
                sBuf = dBuf;
                dBuf = tmpS;
            }
            memcpy(curDst + (size_t)b0 * N,
                   sBuf + (size_t)(b0 - e0) * N,
                   (size_t)(b1 - b0) * N * sizeof(double));
        }
        double * __restrict__ tmpG = curSrc;
        curSrc = curDst;
        curDst = tmpG;
    }

    double end = omp_get_wtime();
    double elapsed_ms = (end - start) * 1000.0;

    printf("Tiempo de ejecucion de Jacobi 2D (%d iters): %.2f ms\n", NUM_ITERS, elapsed_ms);

    double checksum = 0.0;
    for (size_t p = 0; p < (size_t)N * N; ++p) checksum += curSrc[p] * (double)(p % 7 + 1);
    printf("Checksum: %.6f\n", checksum);
    free(A_raw);
    free(B_raw);
    free(t0_raw);
    free(t1_raw);
    return 0;
}