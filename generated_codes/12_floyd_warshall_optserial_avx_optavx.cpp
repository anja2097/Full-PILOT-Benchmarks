/* ANALYSIS: Floyd-Warshall min-plus, N=3072 int.
   Bottleneck: memory bandwidth / load-store ports, row_k re-read N times.
   Transform: cache-block k/i/j (B=256) with phased diag/row/col/remaining order.
   Remaining blocks use k-inner register blocking MR=4 NR=16 (8 acc) to keep C in regs.
   Keeps AVX2 vpaddd/vpminsd, aligned loads/stores, hoisted broadcasts, no prefetch. */
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

    const int B = 256;
    const int MR = 4;
    const int NR = 16;
    int * __restrict__ d = dist;
    const size_t NN_ = static_cast<size_t>(N);

    for (int kk = 0; kk < N; kk += B) {
        // Phase 1: diagonal block (kk,kk), k outer
        for (int k = kk; k < kk + B; ++k) {
            size_t row_k = static_cast<size_t>(k) * NN_;
            for (int i = kk; i < kk + B; ++i) {
                size_t row_i = static_cast<size_t>(i) * NN_;
                __m256i dik_vec = _mm256_set1_epi32(d[row_i + k]);
                for (int j = kk; j < kk + B; j += 8) {
                    __m256i cij = _mm256_load_si256(reinterpret_cast<__m256i const*>(d + row_i + j));
                    __m256i ckj = _mm256_load_si256(reinterpret_cast<__m256i const*>(d + row_k + j));
                    __m256i via = _mm256_add_epi32(dik_vec, ckj);
                    cij = _mm256_min_epi32(cij, via);
                    _mm256_store_si256(reinterpret_cast<__m256i*>(d + row_i + j), cij);
                }
            }
        }
        // Phase 2: row panel (kk, J!=kk)
        for (int k = kk; k < kk + B; ++k) {
            size_t row_k = static_cast<size_t>(k) * NN_;
            for (int i = kk; i < kk + B; ++i) {
                size_t row_i = static_cast<size_t>(i) * NN_;
                __m256i dik_vec = _mm256_set1_epi32(d[row_i + k]);
                for (int J = 0; J < N; J += B) {
                    if (J == kk) continue;
                    for (int j = J; j < J + B; j += 8) {
                        __m256i cij = _mm256_load_si256(reinterpret_cast<__m256i const*>(d + row_i + j));
                        __m256i ckj = _mm256_load_si256(reinterpret_cast<__m256i const*>(d + row_k + j));
                        __m256i via = _mm256_add_epi32(dik_vec, ckj);
                        cij = _mm256_min_epi32(cij, via);
                        _mm256_store_si256(reinterpret_cast<__m256i*>(d + row_i + j), cij);
                    }
                }
            }
        }
        // Phase 2: column panel (I!=kk, kk)
        for (int k = kk; k < kk + B; ++k) {
            size_t row_k = static_cast<size_t>(k) * NN_;
            for (int I = 0; I < N; I += B) {
                if (I == kk) continue;
                for (int i = I; i < I + B; ++i) {
                    size_t row_i = static_cast<size_t>(i) * NN_;
                    __m256i dik_vec = _mm256_set1_epi32(d[row_i + k]);
                    // j in kk block, contiguous B
                    for (int j = kk; j < kk + B; j += 8) {
                        __m256i cij = _mm256_load_si256(reinterpret_cast<__m256i const*>(d + row_i + j));
                        __m256i ckj = _mm256_load_si256(reinterpret_cast<__m256i const*>(d + row_k + j));
                        __m256i via = _mm256_add_epi32(dik_vec, ckj);
                        cij = _mm256_min_epi32(cij, via);
                        _mm256_store_si256(reinterpret_cast<__m256i*>(d + row_i + j), cij);
                    }
                }
            }
        }
        // Phase 3: remaining blocks, k-inner register blocked MR x NR
        for (int JJ = 0; JJ < N; JJ += B) {
            if (JJ == kk) continue;
            for (int II = 0; II < N; II += B) {
                if (II == kk) continue;
                for (int i = II; i < II + B; i += MR) {
                    size_t row0 = static_cast<size_t>(i + 0) * NN_;
                    size_t row1 = static_cast<size_t>(i + 1) * NN_;
                    size_t row2 = static_cast<size_t>(i + 2) * NN_;
                    size_t row3 = static_cast<size_t>(i + 3) * NN_;
                    for (int j = JJ; j < JJ + B; j += NR) {
                        __m256i c00 = _mm256_load_si256(reinterpret_cast<__m256i const*>(d + row0 + j));
                        __m256i c01 = _mm256_load_si256(reinterpret_cast<__m256i const*>(d + row0 + j + 8));
                        __m256i c10 = _mm256_load_si256(reinterpret_cast<__m256i const*>(d + row1 + j));
                        __m256i c11 = _mm256_load_si256(reinterpret_cast<__m256i const*>(d + row1 + j + 8));
                        __m256i c20 = _mm256_load_si256(reinterpret_cast<__m256i const*>(d + row2 + j));
                        __m256i c21 = _mm256_load_si256(reinterpret_cast<__m256i const*>(d + row2 + j + 8));
                        __m256i c30 = _mm256_load_si256(reinterpret_cast<__m256i const*>(d + row3 + j));
                        __m256i c31 = _mm256_load_si256(reinterpret_cast<__m256i const*>(d + row3 + j + 8));
                        for (int k = kk; k < kk + B; ++k) {
                            size_t rk = static_cast<size_t>(k) * NN_;
                            __m256i a0 = _mm256_set1_epi32(d[row0 + k]);
                            __m256i a1 = _mm256_set1_epi32(d[row1 + k]);
                            __m256i a2 = _mm256_set1_epi32(d[row2 + k]);
                            __m256i a3 = _mm256_set1_epi32(d[row3 + k]);
                            __m256i b0 = _mm256_load_si256(reinterpret_cast<__m256i const*>(d + rk + j));
                            __m256i b1 = _mm256_load_si256(reinterpret_cast<__m256i const*>(d + rk + j + 8));
                            c00 = _mm256_min_epi32(c00, _mm256_add_epi32(a0, b0));
                            c01 = _mm256_min_epi32(c01, _mm256_add_epi32(a0, b1));
                            c10 = _mm256_min_epi32(c10, _mm256_add_epi32(a1, b0));
                            c11 = _mm256_min_epi32(c11, _mm256_add_epi32(a1, b1));
                            c20 = _mm256_min_epi32(c20, _mm256_add_epi32(a2, b0));
                            c21 = _mm256_min_epi32(c21, _mm256_add_epi32(a2, b1));
                            c30 = _mm256_min_epi32(c30, _mm256_add_epi32(a3, b0));
                            c31 = _mm256_min_epi32(c31, _mm256_add_epi32(a3, b1));
                        }
                        _mm256_store_si256(reinterpret_cast<__m256i*>(d + row0 + j), c00);
                        _mm256_store_si256(reinterpret_cast<__m256i*>(d + row0 + j + 8), c01);
                        _mm256_store_si256(reinterpret_cast<__m256i*>(d + row1 + j), c10);
                        _mm256_store_si256(reinterpret_cast<__m256i*>(d + row1 + j + 8), c11);
                        _mm256_store_si256(reinterpret_cast<__m256i*>(d + row2 + j), c20);
                        _mm256_store_si256(reinterpret_cast<__m256i*>(d + row2 + j + 8), c21);
                        _mm256_store_si256(reinterpret_cast<__m256i*>(d + row3 + j), c30);
                        _mm256_store_si256(reinterpret_cast<__m256i*>(d + row3 + j + 8), c31);
                    }
                }
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