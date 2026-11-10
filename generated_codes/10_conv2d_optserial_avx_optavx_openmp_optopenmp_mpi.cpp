#include <mpi.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstddef>
#include <immintrin.h>
#include <omp.h>

#define TILE_J 56
#define N 4096
#define R 4
#define W (2 * R + 1)
#define NUM_ITERS 8

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Compute filter */
    double filter[W * W] __attribute__((aligned(64)));
    double fsum = 0.0;
    for (int dy = -R; dy <= R; ++dy) {
        for (int dx = -R; dx <= R; ++dx) {
            double w = std::exp(-(dx * dx + dy * dy) / 8.0);
            filter[(dy + R) * W + (dx + R)] = w;
            fsum += w;
        }
    }
    for (int k = 0; k < W * W; ++k) filter[k] /= fsum;

    /* Pre‑compute offsets for each tap */
    ptrdiff_t offsets[W * W];
    for (int dy = -R; dy <= R; ++dy) {
        for (int dx = -R; dx <= R; ++dx) {
            int k = (dy + R) * W + (dx + R);
            offsets[k] = (ptrdiff_t)dy * N + dx;
        }
    }

    size_t total = (size_t)N * N;
    double* in = (double*)aligned_alloc(64, total * sizeof(double));
    double* out = (double*)aligned_alloc(64, total * sizeof(double));
    if (!in || !out) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* First‑touch initialization */
    #pragma omp parallel for schedule(static) default(none) shared(in, out)
    for (int i = 0; i < N; ++i) {
        double* rin = in + (size_t)i * N;
        double* rout = out + (size_t)i * N;
        for (int j = 0; j < N; ++j) {
            rin[j] = 0.0;
            rout[j] = 0.0;
        }
    }

    srand(42);
    for (size_t p = 0; p < total; ++p) {
        in[p] = static_cast<double>(rand()) / RAND_MAX;
    }
    memcpy(out, in, total * sizeof(double));

    /* Domain decomposition (row wise) */
    int rows_per_rank = N / size;
    int remainder = N % size;
    int local_i_start = rank * rows_per_rank + (rank < remainder ? rank : remainder);
    int local_i_end   = local_i_start + rows_per_rank + (rank < remainder ? 1 : 0);

    int comp_i_start = (local_i_start > R) ? local_i_start : R;
    int comp_i_end   = (local_i_end < (N - R)) ? local_i_end : (N - R);

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    double* lcur = in;
    double* lnxt = out;

    for (int iter = 0; iter < NUM_ITERS; ++iter) {
        const double* __restrict__ cin = lcur;
        double* __restrict__ cout = lnxt;

        #pragma omp parallel for schedule(static) default(none) shared(cin, cout, comp_i_start, comp_i_end, filter, offsets)
        for (int i = comp_i_start; i < comp_i_end; ++i) {
            for (int jj = R; jj < N - R; jj += TILE_J) {
                const double* base = cin + (size_t)i * N + jj;
                double* outrow = cout + (size_t)i * N + jj;

                constexpr int VEC_W = 4;
                constexpr int NB = TILE_J / VEC_W;
                __m256d acc[NB];

                {
                    double w0 = filter[0];
                    __m256d w0v = _mm256_set1_pd(w0);
                    const double* row0 = base + offsets[0];
                    for (int b = 0; b < NB; ++b) {
                        __m256d vec = _mm256_loadu_pd(row0 + b * VEC_W);
                        acc[b] = _mm256_mul_pd(w0v, vec);
                    }
                }

                for (int k = 1; k < W * W; ++k) {
                    double wk = filter[k];
                    __m256d wkv = _mm256_set1_pd(wk);
                    const double* rowk = base + offsets[k];
                    for (int b = 0; b < NB; ++b) {
                        __m256d vec = _mm256_loadu_pd(rowk + b * VEC_W);
                        acc[b] = _mm256_fmadd_pd(wkv, vec, acc[b]);
                    }
                }

                for (int b = 0; b < NB; ++b) {
                    _mm256_storeu_pd(outrow + b * VEC_W, acc[b]);
                }
            }
        }

        /* Swap buffers */
        double* tmp = lcur;
        lcur = lnxt;
        lnxt = tmp;

        /* Exchange halo rows with neighbors (except after final iteration) */
        if (iter != NUM_ITERS - 1) {
            int top    = rank - 1;
            int bottom = rank + 1;

            if (top >= 0) {
                MPI_Sendrecv(lcur + (size_t)local_i_start * N, R * N, MPI_DOUBLE, top, 0,
                             lcur + (size_t)(local_i_start - R) * N, R * N, MPI_DOUBLE, top, 1,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            if (bottom < size) {
                MPI_Sendrecv(lcur + (size_t)(local_i_end - R) * N, R * N, MPI_DOUBLE, bottom, 1,
                             lcur + (size_t)local_i_end * N, R * N, MPI_DOUBLE, bottom, 0,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    }

    double end = MPI_Wtime();
    double elapsed = (end - start) * 1000.0;   /* milliseconds */

    double max_elapsed_ms = 0.0;
    MPI_Reduce(&elapsed, &max_elapsed_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Tiempo de ejecucion de la convolucion 2D (%d iters): %.2f ms\n", NUM_ITERS, max_elapsed_ms);
    }

    /* Compute checksum */
    double* cur = (NUM_ITERS % 2 == 0) ? in : out;
    double local_checksum = 0.0;
    for (int i = local_i_start; i < local_i_end; ++i) {
        size_t base = (size_t)i * N;
        for (size_t j = 0; j < (size_t)N; ++j) {
            size_t p = base + j;
            local_checksum += cur[p] * static_cast<double>(p % 11 + 1);
        }
    }

    double global_checksum = 0.0;
    MPI_Reduce(&local_checksum, &global_checksum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Checksum: %.6f\n", global_checksum);
    }

    free(in);
    free(out);
    MPI_Finalize();
    return 0;
}