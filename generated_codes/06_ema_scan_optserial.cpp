/* ANALYSIS: 
 * Kernel: EMA (exponential moving average) IIR filter applied in-place 40 times.
 * Bottleneck: Serial dependency chain in the recurrence s = ALPHA*s + BETA*x[i].
 * The original loop is latency-bound (~7 cycles/element without FMA).
 * Transformation: Cache-blocked, multi-chain blocked scan.
 * - Divide the array into tiles (32768 elements) to keep working set in L2.
 * - Within each tile, process K=4 blocks simultaneously using an interleaved temporary buffer.
 * - Local scan computes the recurrence with zero initial state for each block, exposing K independent chains.
 * - Sequential correction applies the block-to-block carry with a single dependency chain of length K.
 * - Packing/scattering keeps the hot loops contiguous and vectorizable.
 * - Uses 64-byte aligned memory for the main array.
 * Expected: Convert latency-bound to throughput-bound, enabling SIMD in later stage.
 */
#include <chrono>
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

    for (int iter = 0; iter < NUM_ITERS; ++iter) {
        double s_prev = s0;
        for (long tile = 0; tile < N / TILE_SIZE; ++tile) {
            long tile_start = tile * TILE_SIZE;
            for (long g = 0; g < (TILE_SIZE / BLOCK_SIZE) / K; ++g) {
                long base_block = g * K;

                // Pack K blocks into interleaved temp
                for (int k = 0; k < K; ++k) {
                    long block_start = tile_start + (base_block + k) * BLOCK_SIZE;
                    for (int j = 0; j < BLOCK_SIZE; ++j) {
                        temp[j * K + k] = x[block_start + j];
                    }
                }

                // Local scan on K independent chains
                double t[K] = {0.0};
                for (int j = 0; j < BLOCK_SIZE; ++j) {
                    for (int k = 0; k < K; ++k) {
                        t[k] = alpha * t[k] + beta * temp[j * K + k];
                        temp[j * K + k] = t[k];
                    }
                }

                // Compute block start states and update s_prev
                double s_start_arr[K];
                double s = s_prev;
                for (int k = 0; k < K; ++k) {
                    s_start_arr[k] = s;
                    double t_end = temp[(BLOCK_SIZE - 1) * K + k];
                    s = t_end + alpha_pow_block * s;
                }
                s_prev = s;

                // Correction
                double corr[K];
                for (int k = 0; k < K; ++k) {
                    corr[k] = alpha * s_start_arr[k];
                }
                for (int j = 0; j < BLOCK_SIZE; ++j) {
                    for (int k = 0; k < K; ++k) {
                        temp[j * K + k] += corr[k];
                        corr[k] *= alpha;
                    }
                }

                // Scatter back to x
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