#include <cstdio>
#include <cstdlib>
#include <mpi.h>
#include <omp.h>

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const long N = 1L << 26;
    const int NUM_BINS = 4096;
    const int NUM_ITERS = 50;

    int* __restrict__ data;
    int* __restrict__ hist;

    if (posix_memalign((void**)&data, 64, N * sizeof(int)) != 0) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (posix_memalign((void**)&hist, 64, NUM_BINS * sizeof(int)) != 0) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // First-touch for data: place pages in parallel with same mapping as timed kernel
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < N; ++i) {
        data[i] = 0;
    }

    srand(42);
    for (long i = 0; i < N; ++i) {
        data[i] = rand() % NUM_BINS;
    }

    // First-touch for hist: place pages in parallel with same mapping as timed kernel
    #pragma omp parallel for schedule(static)
    for (int b = 0; b < NUM_BINS; ++b) {
        hist[b] = 0;
    }

    // Domain decomposition across MPI ranks
    long base = N / size;
    long rem  = N % size;
    long local_start = rank * base + (rank < rem ? rank : rem);
    long local_end   = local_start + base + (rank < rem ? 1 : 0);

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    #pragma omp parallel for reduction(+:hist[0:NUM_BINS]) schedule(static)
    for (long i = local_start; i < local_end; ++i) {
        int d = data[i];
        #pragma GCC unroll 50
        for (int iter = 0; iter < NUM_ITERS; ++iter) {
            int bin = (d + iter * 7) & (NUM_BINS - 1);
            hist[bin]++;
        }
    }

    double end = MPI_Wtime();
    double elapsed_ms = (end - start) * 1000.0;

    double max_elapsed_ms = 0.0;
    MPI_Reduce(&elapsed_ms, &max_elapsed_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    int* full_hist = nullptr;
    if (rank == 0) {
        if (posix_memalign((void**)&full_hist, 64, NUM_BINS * sizeof(int)) != 0) {
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    } else {
        // Allocate temporary buffer for reduction
        if (posix_memalign((void**)&full_hist, 64, NUM_BINS * sizeof(int)) != 0) {
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Reduce(hist, full_hist, NUM_BINS, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Tiempo de ejecucion del histograma (%d iters): %.2f ms\n", NUM_ITERS, max_elapsed_ms);
        long long checksum = 0;
        for (int b = 0; b < NUM_BINS; ++b) {
            checksum += (long long)full_hist[b] * (b + 1);
        }
        printf("Checksum: %.6f\n", static_cast<double>(checksum));
    }

    free(data);
    free(hist);
    if (full_hist) free(full_hist);

    MPI_Finalize();
    return 0;
}