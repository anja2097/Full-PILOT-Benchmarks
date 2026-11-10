#include <mpi.h>
#include <omp.h>
#include <immintrin.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>

#ifndef BLOCK_ROWS
#define BLOCK_ROWS 48
#endif
#ifndef TIME_TILE
#define TIME_TILE 8
#endif

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int N = 4096;
    const int NUM_ITERS = 200;

    double *A_raw = nullptr, *B_raw = nullptr;
    (void)posix_memalign((void **)&A_raw, 64, (size_t)N * N * sizeof(double));
    (void)posix_memalign((void **)&B_raw, 64, (size_t)N * N * sizeof(double));
    double *__restrict__ A = A_raw;
    double *__restrict__ B = B_raw;

    const int Hmax = BLOCK_ROWS + 2 * TIME_TILE;
    int maxThr = omp_get_max_threads();
    if (maxThr < 1)
        maxThr = 1;
    size_t perThrElems = (size_t)Hmax * (size_t)N;
    size_t perThrBytes = perThrElems * sizeof(double);
    double *t0_raw = nullptr, *t1_raw = nullptr;
    (void)posix_memalign((void **)&t0_raw, 64, perThrBytes * (size_t)maxThr);
    (void)posix_memalign((void **)&t1_raw, 64, perThrBytes * (size_t)maxThr);
    double *__restrict__ tmp0_big = t0_raw;
    double *__restrict__ tmp1_big = t1_raw;

    // NUMA first-touch with same static mapping as timed kernel, before serial init
#pragma omp parallel for schedule(static) default(none) shared(A, B)
    for (int i = 0; i < N; ++i)
    {
        double *__restrict__ rA = A + (size_t)i * N;
        double *__restrict__ rB = B + (size_t)i * N;
        for (int j = 0; j < N; j += 64)
        {
            rA[j] = 0.0;
            rB[j] = 0.0;
        }
    }
#pragma omp parallel default(none) shared(tmp0_big, tmp1_big, perThrElems, maxThr)
    {
        int tid = omp_get_thread_num();
        if (tid < maxThr)
        {
            double *p0 = tmp0_big + (size_t)tid * perThrElems;
            double *p1 = tmp1_big + (size_t)tid * perThrElems;
            memset(p0, 0, perThrElems * sizeof(double));
            memset(p1, 0, perThrElems * sizeof(double));
        }
    }

    srand(42);
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            bool border = (i == 0 || j == 0 || i == N - 1 || j == N - 1);
            A[(size_t)i * N + j] = border ? 1.0 : (double)rand() / RAND_MAX;
        }
    }
    memcpy(B, A, (size_t)N * N * sizeof(double));

    double *__restrict__ curSrc = A;
    double *__restrict__ curDst = B;

    // Domain decomposition: rows 1 .. N-2 are divided among ranks
    int totalInterior = N - 2;
    int rowsPerRank = totalInterior / size;
    int remainder = totalInterior % size;
    std::vector<int> rows_local(size);
    std::vector<int> recvcounts(size);
    std::vector<int> displs(size);
    int offsetRows = 1; // start after the first border row
    for (int r = 0; r < size; ++r)
    {
        rows_local[r] = rowsPerRank + (r < remainder ? 1 : 0);
        recvcounts[r] = rows_local[r] * N;
        displs[r] = offsetRows * N;
        offsetRows += rows_local[r];
    }
    int local_start = 1;
    for (int r = 0; r < rank; ++r)
        local_start += rows_local[r];
    int local_rows = rows_local[rank];
    int local_end = local_start + local_rows; // exclusive

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    for (int t0 = 0; t0 < NUM_ITERS; t0 += TIME_TILE)
    {
        int Tcur = std::min(TIME_TILE, NUM_ITERS - t0);

#pragma omp parallel default(none) shared(curSrc, curDst, tmp0_big, tmp1_big, perThrElems, maxThr, Tcur, local_start, local_end)
        {
            int tid = omp_get_thread_num();
            double *__restrict__ myT0 = tmp0_big + (size_t)tid * perThrElems;
            double *__restrict__ myT1 = tmp1_big + (size_t)tid * perThrElems;
            const __m256d coeff = _mm256_set1_pd(0.2);
#pragma omp for schedule(static)
            for (int b0 = local_start; b0 < local_end; b0 += BLOCK_ROWS)
            {
                int b1 = std::min(b0 + BLOCK_ROWS, N - 1);
                int e0 = b0 - Tcur;
                if (e0 < 0)
                    e0 = 0;
                int e1 = b1 + Tcur;
                if (e1 > N)
                    e1 = N;
                int H = e1 - e0;
                double *__restrict__ tmp0 = myT0;
                double *__restrict__ tmp1 = myT1;
                memcpy(tmp0, curSrc + (size_t)e0 * N, (size_t)H * N * sizeof(double));
                memcpy(tmp1, curSrc + (size_t)e0 * N, (size_t)H * N * sizeof(double));

                double *__restrict__ sBuf = tmp0;
                double *__restrict__ dBuf = tmp1;
                for (int s = 0; s < Tcur; ++s)
                {
                    int c0 = b0 - (Tcur - 1 - s);
                    if (c0 < 1)
                        c0 = 1;
                    int c1 = b1 + (Tcur - 1 - s);
                    if (c1 > N - 1)
                        c1 = N - 1;

                    for (int i = c0; i < c1; ++i)
                    {
                        size_t li = (size_t)(i - e0);
                        double *__restrict__ dRow = dBuf + li * N;
                        const double *__restrict__ sRow = sBuf + li * N;
                        const double *__restrict__ sRowM = sRow - N;
                        const double *__restrict__ sRowP = sRow + N;
                        dRow[0] = sRow[0];
                        dRow[N - 1] = sRow[N - 1];

                        int j = 1;
                        int vec_end = N - 5;
                        for (; j <= vec_end; j += 4)
                        {
                            __m256d s_center = _mm256_loadu_pd(&sRow[j]);
                            __m256d s_left = _mm256_loadu_pd(&sRow[j - 1]);
                            __m256d s_right = _mm256_loadu_pd(&sRow[j + 1]);
                            __m256d s_up = _mm256_loadu_pd(&sRowM[j]);
                            __m256d s_down = _mm256_loadu_pd(&sRowP[j]);

                            __m256d sum = _mm256_add_pd(s_center, s_left);
                            sum = _mm256_add_pd(sum, s_right);
                            sum = _mm256_add_pd(sum, s_up);
                            sum = _mm256_add_pd(sum, s_down);
                            __m256d res = _mm256_mul_pd(coeff, sum);
                            _mm256_storeu_pd(&dRow[j], res);
                        }
                        for (; j < N - 1; ++j)
                        {
                            dRow[j] = 0.2 *
                                      (sRow[j] + sRow[j - 1] + sRow[j + 1] +
                                       sRowM[j] + sRowP[j]);
                        }
                    }

                    for (int i = e0; i < c0; ++i)
                    {
                        size_t li = (size_t)(i - e0);
                        memcpy(dBuf + li * N, sBuf + li * N, (size_t)N * sizeof(double));
                    }
                    for (int i = c1; i < e1; ++i)
                    {
                        size_t li = (size_t)(i - e0);
                        memcpy(dBuf + li * N, sBuf + li * N, (size_t)N * sizeof(double));
                    }
                    double *__restrict__ tmpS = sBuf;
                    sBuf = dBuf;
                    dBuf = tmpS;
                }
                memcpy(curDst + (size_t)b0 * N,
                       sBuf + (size_t)(b0 - e0) * N,
                       (size_t)(b1 - b0) * N * sizeof(double));
            }
        }

        // Swap source and destination buffers
        double *__restrict__ tmpPtr = curSrc;
        curSrc = curDst;
        curDst = tmpPtr;

        // Gather the updated interior rows from all ranks into curDst (which now holds the full grid)
        MPI_Allgatherv(curSrc + (size_t)local_start * N,
                       local_rows * N,
                       MPI_DOUBLE,
                       curDst,
                       recvcounts.data(),
                       displs.data(),
                       MPI_DOUBLE,
                       MPI_COMM_WORLD);

        // After gathering, make curSrc point to the full grid for the next tile
        tmpPtr = curSrc;
        curSrc = curDst;
        curDst = tmpPtr;
    }

    double end = MPI_Wtime();
    double elapsed_ms = (end - start) * 1000.0;

    double max_elapsed_ms = 0.0;
    MPI_Reduce(&elapsed_ms, &max_elapsed_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        printf("Tiempo de ejecucion de Jacobi 2D (%d iters): %.2f ms\n", NUM_ITERS, max_elapsed_ms);
    }

    double checksum = 0.0;
    for (size_t p = 0; p < (size_t)N * N; ++p)
        checksum += curSrc[p] * (double)(p % 7 + 1);

    double global_checksum = 0.0;
    MPI_Reduce(&checksum, &global_checksum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0)
    {
        printf("Checksum: %.6f\n", global_checksum);
    }

    free(A_raw);
    free(B_raw);
    free(t0_raw);
    free(t1_raw);
    MPI_Finalize();
    return 0;
}