#include <mpi.h>
#include <omp.h>
#include <immintrin.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>

#define N (1L << 25)
#define NUM_ITERS 40
#define ALPHA 0.999
#define BETA (1.0 - ALPHA)

#define TILE_SIZE 32768
#define BLOCK_SIZE 512
#define K 4

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const long NTILES = N / TILE_SIZE;          // 1024
    long base_tiles = NTILES / size;
    long rem_tiles = NTILES % size;
    long local_tile_start = rank * base_tiles + (rank < rem_tiles ? rank : rem_tiles);
    long local_tile_count = base_tiles + (rank < rem_tiles ? 1 : 0);
    long local_elem_start = local_tile_start * TILE_SIZE;
    long local_N = local_tile_count * TILE_SIZE;

    double* x_local = nullptr;
    if (posix_memalign((void**)&x_local, 64, local_N * sizeof(double)) != 0) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Allocate wpow (same for all ranks)
    double* wpow = nullptr;
    if (posix_memalign((void**)&wpow, 64, TILE_SIZE * sizeof(double)) != 0) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Allocate local tile_add and s_in
    double* tile_add_local = nullptr;
    double* s_in_local = nullptr;
    if (posix_memalign((void**)&tile_add_local, 64, local_tile_count * sizeof(double)) != 0) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (posix_memalign((void**)&s_in_local, 64, local_tile_count * sizeof(double)) != 0) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Generate full x on rank 0 and scatter to all ranks
    double* x_full = nullptr;
    if (rank == 0) {
        if (posix_memalign((void**)&x_full, 64, N * sizeof(double)) != 0) {
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        for (long i = 0; i < N; ++i) x_full[i] = 0.0;
        srand(42);
        for (long i = 0; i < N; ++i) {
            x_full[i] = static_cast<double>(rand()) / RAND_MAX;
        }
    }

    // Prepare counts and displacements for scatterv of x
    int* sendcounts_x = nullptr;
    int* displs_x = nullptr;
    if (rank == 0) {
        sendcounts_x = (int*)malloc(size * sizeof(int));
        displs_x = (int*)malloc(size * sizeof(int));
        for (int r = 0; r < size; ++r) {
            long r_base_tiles = NTILES / size;
            long r_rem_tiles = NTILES % size;
            long r_tile_start = r * r_base_tiles + (r < r_rem_tiles ? r : r_rem_tiles);
            long r_tile_cnt = r_base_tiles + (r < r_rem_tiles ? 1 : 0);
            sendcounts_x[r] = (int)(r_tile_cnt * TILE_SIZE);
            displs_x[r] = (int)(r_tile_start * TILE_SIZE);
        }
    }

    MPI_Scatterv(x_full, sendcounts_x, displs_x, MPI_DOUBLE,
                 x_local, (int)local_N, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    // Broadcast wpow construction
    const double alpha = ALPHA;
    const double beta = BETA;
    const double alpha_pow_block = pow(alpha, BLOCK_SIZE);
    double s0 = 0.0;
    if (rank == 0) {
        s0 = x_full[0];
    }
    MPI_Bcast(&s0, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Build wpow (same on all ranks)
    {
        double p = 1.0;
        for (long j = TILE_SIZE - 1; j >= 0; --j) {
            wpow[j] = BETA * p;
            p *= ALPHA;
        }
    }

    double alpha_T = pow(alpha, (double)TILE_SIZE);

    // Allocate full buffers on root for gather/scatter of tile_add and s_in
    double* tile_add_full = nullptr;
    double* s_in_full = nullptr;
    if (rank == 0) {
        tile_add_full = (double*)malloc(NTILES * sizeof(double));
        s_in_full = (double*)malloc(NTILES * sizeof(double));
    }

    // Prepare counts and displs for gather/scatter of tile arrays
    int* tile_counts = nullptr;
    int* tile_displs = nullptr;
    if (rank == 0) {
        tile_counts = (int*)malloc(size * sizeof(int));
        tile_displs = (int*)malloc(size * sizeof(int));
        for (int r = 0; r < size; ++r) {
            long r_base_tiles = NTILES / size;
            long r_rem_tiles = NTILES % size;
            long r_tile_start = r * r_base_tiles + (r < r_rem_tiles ? r : r_rem_tiles);
            long r_tile_cnt = r_base_tiles + (r < r_rem_tiles ? 1 : 0);
            tile_counts[r] = (int)r_tile_cnt;
            tile_displs[r] = (int)r_tile_start;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    const __m256d alpha_v = _mm256_set1_pd(alpha);
    const __m256d beta_v  = _mm256_set1_pd(beta);

    for (int iter = 0; iter < NUM_ITERS; ++iter) {
        // Compute tile_add for local tiles
        #pragma omp parallel for schedule(static,8) default(none) \
            shared(x_local, tile_add_local, wpow, local_tile_count) \
            firstprivate(alpha_v, beta_v)
        for (long t = 0; t < local_tile_count; ++t) {
            double* __restrict__ xs = x_local + t * TILE_SIZE;
            double* __restrict__ ww = wpow;
            double a0 = 0.0, a1 = 0.0, a2 = 0.0, a3 = 0.0;
            for (long j = 0; j < TILE_SIZE; j += 4) {
                a0 += xs[j + 0] * ww[j + 0];
                a1 += xs[j + 1] * ww[j + 1];
                a2 += xs[j + 2] * ww[j + 2];
                a3 += xs[j + 3] * ww[j + 3];
            }
            tile_add_local[t] = (a0 + a1) + (a2 + a3);
        }

        // Gather tile_add to root
        MPI_Gatherv(tile_add_local, (int)local_tile_count, MPI_DOUBLE,
                    tile_add_full, tile_counts, tile_displs, MPI_DOUBLE,
                    0, MPI_COMM_WORLD);

        // Root computes full s_in array
        if (rank == 0) {
            double s = s0;
            for (long t = 0; t < NTILES; ++t) {
                s_in_full[t] = s;
                s = tile_add_full[t] + alpha_T * s;
            }
        }

        // Scatter s_in back to ranks
        MPI_Scatterv(s_in_full, tile_counts, tile_displs, MPI_DOUBLE,
                     s_in_local, (int)local_tile_count, MPI_DOUBLE,
                     0, MPI_COMM_WORLD);

        // Update x using s_in_local
        #pragma omp parallel for schedule(static,8) default(none) \
            shared(x_local, s_in_local, wpow, alpha_T, local_tile_count) \
            firstprivate(alpha_v, beta_v)
        for (long t = 0; t < local_tile_count; ++t) {
            double s_prev = s_in_local[t];
            alignas(64) double temp[K * BLOCK_SIZE];
            long tile_start = t * TILE_SIZE;
            for (long g = 0; g < (TILE_SIZE / BLOCK_SIZE) / K; ++g) {
                long base_block = g * K;
                double* p0 = x_local + tile_start + (base_block + 0) * BLOCK_SIZE;
                double* p1 = x_local + tile_start + (base_block + 1) * BLOCK_SIZE;
                double* p2 = x_local + tile_start + (base_block + 2) * BLOCK_SIZE;
                double* p3 = x_local + tile_start + (base_block + 3) * BLOCK_SIZE;

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

    double end = MPI_Wtime();
    double elapsed_ms = (end - start) * 1000.0;

    double max_elapsed_ms = 0.0;
    MPI_Reduce(&elapsed_ms, &max_elapsed_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Compute checksum locally
    double local_checksum = 0.0;
    for (long i = 0; i < local_N; ++i) {
        long global_idx = local_elem_start + i;
        local_checksum += x_local[i] * static_cast<double>((global_idx % 13) + 1);
    }

    double checksum = 0.0;
    MPI_Reduce(&local_checksum, &checksum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Tiempo de ejecucion del filtro EMA (%d iters): %.2f ms\n", NUM_ITERS, max_elapsed_ms);
        printf("Checksum: %.6f\n", checksum);
    }

    // Clean up
    free(x_local);
    free(wpow);
    free(tile_add_local);
    free(s_in_local);
    if (rank == 0) {
        free(x_full);
        free(tile_add_full);
        free(s_in_full);
        free(sendcounts_x);
        free(displs_x);
        free(tile_counts);
        free(tile_displs);
    }

    MPI_Finalize();
    return 0;
}