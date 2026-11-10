/* ANALYSIS: 
 * Kernel: B = 0.5*B + A^T, repeated NUM_ITERS times.
 * Bottleneck: Original has strided B access causing TLB thrashing and poor cache use.
 * Transformations:
 * - Tiled loops over i and j to fit A and B blocks in L1 (TILE_SIZE=32, 8KB per block).
 * - Loaded A block into a transposed temp buffer so the hot loop has contiguous B and temp accesses.
 * - Kept iter loop outer over the tile, enabling auto-vectorization of the inner i loop.
 * - Used posix_memalign for 64-byte alignment of A and B.
 * - Result: DRAM traffic reduced to ~864MB total, fully vectorizable inner loop.
 */
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifndef TILE_SIZE
#define TILE_SIZE 32
#endif

int main()
{
    const int N = 6144;
    const int NUM_ITERS = 10;

    const size_t total_size = static_cast<size_t>(N) * N;
    const size_t size_bytes = total_size * sizeof(double);

    double * __restrict__ A;
    double * __restrict__ B;
    if (posix_memalign((void**)&A, 64, size_bytes) != 0) return 1;
    if (posix_memalign((void**)&B, 64, size_bytes) != 0) return 1;

    memset(B, 0, size_bytes);

    srand(42);
    for (size_t p = 0; p < total_size; ++p) {
        A[p] = static_cast<double>(rand()) / RAND_MAX;
    }

    auto start = std::chrono::high_resolution_clock::now();

    alignas(64) double temp[TILE_SIZE][TILE_SIZE];

    for (int i0 = 0; i0 < N; i0 += TILE_SIZE) {
        for (int j0 = 0; j0 < N; j0 += TILE_SIZE) {
            // Transpose A block into temp
            for (int i = 0; i < TILE_SIZE; ++i) {
                const double* A_row = &A[static_cast<size_t>(i0 + i) * N + j0];
                for (int j = 0; j < TILE_SIZE; ++j) {
                    temp[j][i] = A_row[j];
                }
            }

            // Hot loop: 10 iterations over B block
            for (int iter = 0; iter < NUM_ITERS; ++iter) {
                for (int j = 0; j < TILE_SIZE; ++j) {
                    double* __restrict__ B_row = &B[static_cast<size_t>(j0 + j) * N + i0];
                    const double* __restrict__ temp_row = temp[j];
                    for (int i = 0; i < TILE_SIZE; ++i) {
                        B_row[i] = 0.5 * B_row[i] + temp_row[i];
                    }
                }
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    printf("Tiempo de ejecucion de la trasposicion (%d iters): %.2f ms\n", NUM_ITERS, elapsed.count());

    double checksum = 0.0;
    for (size_t p = 0; p < total_size; ++p) {
        checksum += B[p] * static_cast<double>(p % 17 + 1);
    }
    printf("Checksum: %.6f\n", checksum);

    free(A);
    free(B);
    return 0;
}