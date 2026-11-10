/* ANALYSIS: integer Floyd‑Warshall, memory‑bound, vectorized inner j‑loop with AVX2 256‑bit integer intrinsics, aligned allocation, scalar tails handled. */
#include <immintrin.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstddef>

int main()
{
    const int N = 3072;
    const size_t NN = static_cast<size_t>(N) * N;

    // aligned allocation for AVX2 (32‑byte)
    int *dist = nullptr;
    if (posix_memalign(reinterpret_cast<void **>(&dist), 32, NN * sizeof(int)) != 0) {
        std::perror("posix_memalign");
        return 1;
    }

    std::srand(42);
    for (int i = 0; i < N; ++i) {
        size_t row = static_cast<size_t>(i) * N;
        for (int j = 0; j < N; ++j) {
            dist[row + j] = (i == j) ? 0 : 1 + std::rand() % 1000;
        }
    }

    auto start = std::chrono::high_resolution_clock::now();

    const int VEC = 8; // AVX2 processes 8 ints per vector
    for (int k = 0; k < N; ++k) {
        size_t row_k = static_cast<size_t>(k) * N;
        for (int i = 0; i < N; ++i) {
            size_t row_i = static_cast<size_t>(i) * N;
            int dik = dist[row_i + k];
            __m256i dik_vec = _mm256_set1_epi32(dik);

            size_t j = 0;
            // vectorized part
            for (; j + VEC - 1 < static_cast<size_t>(N); j += VEC) {
                __m256i vec_i_j  = _mm256_load_si256(reinterpret_cast<__m256i const*>(dist + row_i + j));
                __m256i vec_k_j  = _mm256_load_si256(reinterpret_cast<__m256i const*>(dist + row_k + j));
                __m256i via_vec  = _mm256_add_epi32(dik_vec, vec_k_j);
                __m256i min_vec  = _mm256_min_epi32(vec_i_j, via_vec);
                _mm256_store_si256(reinterpret_cast<__m256i*>(dist + row_i + j), min_vec);
            }
            // scalar tail
            for (; j < static_cast<size_t>(N); ++j) {
                int via = dik + dist[row_k + j];
                int &dij = dist[row_i + j];
                dij = std::min(dij, via);
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    std::printf("Tiempo de ejecucion de Floyd-Warshall (N=%d): %.2f ms\n", N, elapsed.count());

    long long checksum = 0;
    for (int i = 0; i < N; ++i) {
        size_t row = static_cast<size_t>(i) * N;
        for (int j = 0; j < N; ++j) {
            checksum += static_cast<long long>(dist[row + j]) * ((i + j) % 7 + 1);
        }
    }
    std::printf("Checksum: %.6f\n", static_cast<double>(checksum));

    std::free(dist);
    return 0;
}