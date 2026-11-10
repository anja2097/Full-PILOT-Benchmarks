/* ANALYSIS: Black-Scholes kernel, compute‑bound with modest memory traffic.
 * Vectorization: AVX2 (4×double). Outer option loop is SIMD‑ized; scalar libm
 * calls (log, exp, cnd) are performed per lane, spilled to temporaries and
 * reloaded. Accumulators are kept in vector registers, reducing loads/stores.
 */

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <immintrin.h>

static double cnd(double d)
{
    const double A1 = 0.31938153;
    const double A2 = -0.356563782;
    const double A3 = 1.781477937;
    const double A4 = -1.821255978;
    const double A5 = 1.330274429;
    const double RSQRT2PI = 0.39894228040143267794;

    double k = 1.0 / (1.0 + 0.2316419 * std::fabs(d));
    double poly = k * (A1 + k * (A2 + k * (A3 + k * (A4 + k * A5))));
    double c = RSQRT2PI * std::exp(-0.5 * d * d) * poly;
    return d > 0 ? 1.0 - c : c;
}

int main()
{
    const int N = 1 << 22;            // 4.2 M opciones
    const int NUM_ITERS = 20;

    // Allocate arrays with 64-byte alignment
    double* S = (double*)aligned_alloc(64, N * sizeof(double));
    double* K = (double*)aligned_alloc(64, N * sizeof(double));
    double* T = (double*)aligned_alloc(64, N * sizeof(double));
    double* R = (double*)aligned_alloc(64, N * sizeof(double));
    double* V = (double*)aligned_alloc(64, N * sizeof(double));
    double* call_acc = (double*)aligned_alloc(64, N * sizeof(double));
    double* put_acc = (double*)aligned_alloc(64, N * sizeof(double));

    // Check allocations
    if (!S || !K || !T || !R || !V || !call_acc || !put_acc) {
        fprintf(stderr, "Memory allocation failed\n");
        free(S); free(K); free(T); free(R); free(V); free(call_acc); free(put_acc);
        return 1;
    }

    srand(42);
    for (int i = 0; i < N; ++i) {
        S[i] = 10.0 + 100.0 * static_cast<double>(rand()) / RAND_MAX;
        K[i] = 10.0 + 100.0 * static_cast<double>(rand()) / RAND_MAX;
        T[i] = 0.25 + 2.0 * static_cast<double>(rand()) / RAND_MAX;
        R[i] = 0.01 + 0.05 * static_cast<double>(rand()) / RAND_MAX;
        V[i] = 0.10 + 0.40 * static_cast<double>(rand()) / RAND_MAX;
    }

    auto start = std::chrono::high_resolution_clock::now();

    const int vec_width = 4;
    int i = 0;
    for (; i <= N - vec_width; i += vec_width) {
        // Load contiguous input vectors
        __m256d s_vec = _mm256_load_pd(&S[i]);
        __m256d k_vec = _mm256_load_pd(&K[i]);
        __m256d t_vec = _mm256_load_pd(&T[i]);
        __m256d r_vec = _mm256_load_pd(&R[i]);
        __m256d v_vec = _mm256_load_pd(&V[i]);

        // ---- invariant per‑option terms (scalar libm for log) ----
        // log(s/k) per lane
        __m256d s_div_k = _mm256_div_pd(s_vec, k_vec);
        double s_div_k_arr[vec_width];
        _mm256_storeu_pd(s_div_k_arr, s_div_k);
        double log_sk_arr[vec_width];
        for (int lane = 0; lane < vec_width; ++lane)
            log_sk_arr[lane] = std::log(s_div_k_arr[lane]);
        __m256d log_sk_vec = _mm256_loadu_pd(log_sk_arr);

        // other invariant vectors
        __m256d v_squared      = _mm256_mul_pd(v_vec, v_vec);
        __m256d half_v_squared = _mm256_mul_pd(_mm256_set1_pd(0.5), v_squared);
        __m256d sqrt_t         = _mm256_sqrt_pd(t_vec);
        __m256d v_sqrt_t       = _mm256_mul_pd(v_vec, sqrt_t);   // denominator

        // accumulators
        __m256d call_acc_vec = _mm256_setzero_pd();
        __m256d put_acc_vec  = _mm256_setzero_pd();

        // ---- inner iteration loop (scalar libm per lane) ----
        for (int iter = 0; iter < NUM_ITERS; ++iter) {
            // r = r_val + 1e-4 * iter (vector)
            __m256d r_iter = _mm256_add_pd(r_vec,
                               _mm256_set1_pd(1e-4 * static_cast<double>(iter)));

            // d1 = (log_sk + (r + half_v_squared) * t) / v_sqrt_t
            __m256d r_plus_hvs = _mm256_add_pd(r_iter, half_v_squared);
            __m256d mul_r_t    = _mm256_mul_pd(r_plus_hvs, t_vec);
            __m256d num        = _mm256_add_pd(log_sk_vec, mul_r_t);
            __m256d d1_vec     = _mm256_div_pd(num, v_sqrt_t);
            __m256d d2_vec     = _mm256_sub_pd(d1_vec, v_sqrt_t);

            // disc = k * exp(-r * t)   (scalar exp per lane)
            double r_arr[vec_width];
            _mm256_storeu_pd(r_arr, r_iter);
            double disc_arr[vec_width];
            for (int lane = 0; lane < vec_width; ++lane) {
                disc_arr[lane] = K[i + lane] * std::exp(-r_arr[lane] * T[i + lane]);
            }
            __m256d disc_vec = _mm256_loadu_pd(disc_arr);

            // compute cnd for needed arguments (scalar per lane)
            double d1_arr[vec_width];
            double d2_arr[vec_width];
            _mm256_storeu_pd(d1_arr, d1_vec);
            _mm256_storeu_pd(d2_arr, d2_vec);

            double cnd_d1_arr[vec_width];
            double cnd_m_d1_arr[vec_width];
            double cnd_d2_arr[vec_width];
            double cnd_m_d2_arr[vec_width];
            for (int lane = 0; lane < vec_width; ++lane) {
                cnd_d1_arr[lane]   = cnd(d1_arr[lane]);
                cnd_m_d1_arr[lane] = cnd(-d1_arr[lane]);
                cnd_d2_arr[lane]   = cnd(d2_arr[lane]);
                cnd_m_d2_arr[lane] = cnd(-d2_arr[lane]);
            }
            __m256d cnd_d1_vec   = _mm256_loadu_pd(cnd_d1_arr);
            __m256d cnd_m_d1_vec = _mm256_loadu_pd(cnd_m_d1_arr);
            __m256d cnd_d2_vec   = _mm256_loadu_pd(cnd_d2_arr);
            __m256d cnd_m_d2_vec = _mm256_loadu_pd(cnd_m_d2_arr);

            // call = s * cnd(d1) - disc * cnd(d2)
            __m256d term1 = _mm256_mul_pd(s_vec, cnd_d1_vec);
            __m256d term2 = _mm256_mul_pd(disc_vec, cnd_d2_vec);
            __m256d call_vec = _mm256_sub_pd(term1, term2);

            // put = disc * cnd(-d2) - s * cnd(-d1)
            __m256d term3 = _mm256_mul_pd(disc_vec, cnd_m_d2_vec);
            __m256d term4 = _mm256_mul_pd(s_vec, cnd_m_d1_vec);
            __m256d put_vec = _mm256_sub_pd(term3, term4);

            // accumulate
            call_acc_vec = _mm256_add_pd(call_acc_vec, call_vec);
            put_acc_vec  = _mm256_add_pd(put_acc_vec , put_vec);
        }

        // Store results
        _mm256_store_pd(&call_acc[i], call_acc_vec);
        _mm256_store_pd(&put_acc[i],  put_acc_vec);
    }

    // ----- scalar tail -----
    for (; i < N; ++i) {
        double s_val = S[i];
        double k_val = K[i];
        double t_val = T[i];
        double r_val = R[i];
        double v_val = V[i];

        double log_sk = std::log(s_val / k_val);
        double v_squared = v_val * v_val;
        double half_v_squared = 0.5 * v_squared;
        double sqrt_t = std::sqrt(t_val);
        double v_sqrt_t = v_val * sqrt_t; // denominator for d1

        double call_accum = 0.0;
        double put_accum  = 0.0;
        for (int iter = 0; iter < NUM_ITERS; ++iter) {
            double r = r_val + 1e-4 * iter;
            double d1 = (log_sk + (r + half_v_squared) * t_val) / v_sqrt_t;
            double d2 = d1 - v_sqrt_t;
            double disc = k_val * std::exp(-r * t_val);
            double call = s_val * cnd(d1) - disc * cnd(d2);
            double put  = disc * cnd(-d2) - s_val * cnd(-d1);
            call_accum += call;
            put_accum  += put;
        }
        call_acc[i] = call_accum;
        put_acc[i]  = put_accum;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    printf("Tiempo de ejecucion de Black-Scholes (%d iters): %.2f ms\n", NUM_ITERS, elapsed.count());

    double checksum = 0.0;
    for (int i = 0; i < N; ++i) {
        checksum += call_acc[i] + put_acc[i];
    }
    printf("Checksum: %.6f\n", checksum);

    // Free memory
    free(S);
    free(K);
    free(T);
    free(R);
    free(V);
    free(call_acc);
    free(put_acc);

    return 0;
}