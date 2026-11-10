#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <immintrin.h>
#include <omp.h>
#include <mpi.h>

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

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int N = 1 << 22;
    const int NUM_ITERS = 20;
    const int vec_width = 4;

    double* S = (double*)aligned_alloc(64, N * sizeof(double));
    double* K = (double*)aligned_alloc(64, N * sizeof(double));
    double* T = (double*)aligned_alloc(64, N * sizeof(double));
    double* R = (double*)aligned_alloc(64, N * sizeof(double));
    double* V = (double*)aligned_alloc(64, N * sizeof(double));
    double* call_acc = (double*)aligned_alloc(64, N * sizeof(double));
    double* put_acc = (double*)aligned_alloc(64, N * sizeof(double));

    if (!S || !K || !T || !R || !V || !call_acc || !put_acc) {
        if (rank == 0) fprintf(stderr, "Memory allocation failed\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    srand(42);
    for (int i = 0; i < N; ++i) {
        S[i] = 10.0 + 100.0 * static_cast<double>(rand()) / RAND_MAX;
        K[i] = 10.0 + 100.0 * static_cast<double>(rand()) / RAND_MAX;
        T[i] = 0.25 + 2.0 * static_cast<double>(rand()) / RAND_MAX;
        R[i] = 0.01 + 0.05 * static_cast<double>(rand()) / RAND_MAX;
        V[i] = 0.10 + 0.40 * static_cast<double>(rand()) / RAND_MAX;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    const __m256d sign_mask = _mm256_set1_pd(-0.0);
    const __m256d one = _mm256_set1_pd(1.0);
    const __m256d zero = _mm256_setzero_pd();
    const __m256d RSQRT2PI_vec = _mm256_set1_pd(0.39894228040143267794);
    const __m256d A1 = _mm256_set1_pd(0.31938153);
    const __m256d A2 = _mm256_set1_pd(-0.356563782);
    const __m256d A3 = _mm256_set1_pd(1.781477937);
    const __m256d A4 = _mm256_set1_pd(-1.821255978);
    const __m256d A5 = _mm256_set1_pd(1.330274429);
    const __m256d half = _mm256_set1_pd(0.5);
    const __m256d neg_half = _mm256_set1_pd(-0.5);
    const __m256d c2316419 = _mm256_set1_pd(0.2316419);

    auto cnd_vec = [&](__m256d d) -> __m256d {
        __m256d abs_d = _mm256_andnot_pd(sign_mask, d);
        __m256d k = _mm256_div_pd(one, _mm256_fmadd_pd(c2316419, abs_d, one));
        __m256d t0 = _mm256_fmadd_pd(k, A5, A4);
        __m256d t1 = _mm256_fmadd_pd(k, t0, A3);
        __m256d t2 = _mm256_fmadd_pd(k, t1, A2);
        __m256d t3 = _mm256_fmadd_pd(k, t2, A1);
        __m256d poly = _mm256_mul_pd(k, t3);
        __m256d half_d = _mm256_mul_pd(neg_half, d);
        __m256d exp_arg = _mm256_mul_pd(half_d, d);
        double exp_arg_arr[4];
        _mm256_storeu_pd(exp_arg_arr, exp_arg);
        double exp_arr[4];
        for (int l = 0; l < 4; ++l) exp_arr[l] = std::exp(exp_arg_arr[l]);
        __m256d exp_vec = _mm256_loadu_pd(exp_arr);
        __m256d c = _mm256_mul_pd(_mm256_mul_pd(RSQRT2PI_vec, exp_vec), poly);
        __m256d mask = _mm256_cmp_pd(d, zero, _CMP_GT_OQ);
        __m256d one_minus_c = _mm256_sub_pd(one, c);
        return _mm256_blendv_pd(c, one_minus_c, mask);
    };

    int base = N / size;
    int rem = N % size;
    int local_start = rank * base + (rank < rem ? rank : rem);
    int local_n = base + (rank < rem ? 1 : 0);
    int local_end = local_start + local_n;

    #pragma omp parallel for schedule(static) default(none) \
        shared(S, K, T, R, V, call_acc, put_acc, local_start, local_end) \
        firstprivate(cnd_vec)
    for (int i = local_start; i <= local_end - vec_width; i += vec_width) {
        __m256d s_vec = _mm256_load_pd(&S[i]);
        __m256d k_vec = _mm256_load_pd(&K[i]);
        __m256d t_vec = _mm256_load_pd(&T[i]);
        __m256d r_vec = _mm256_load_pd(&R[i]);
        __m256d v_vec = _mm256_load_pd(&V[i]);

        __m256d s_div_k = _mm256_div_pd(s_vec, k_vec);
        double s_div_k_arr[4];
        _mm256_storeu_pd(s_div_k_arr, s_div_k);
        double log_sk_arr[4];
        for (int l = 0; l < 4; ++l) log_sk_arr[l] = std::log(s_div_k_arr[l]);
        __m256d log_sk_vec = _mm256_loadu_pd(log_sk_arr);

        __m256d v_squared = _mm256_mul_pd(v_vec, v_vec);
        __m256d half_v_squared = _mm256_mul_pd(half, v_squared);
        __m256d sqrt_t = _mm256_sqrt_pd(t_vec);
        __m256d v_sqrt_t = _mm256_mul_pd(v_vec, sqrt_t);

        __m256d call_acc_vec = _mm256_setzero_pd();
        __m256d put_acc_vec  = _mm256_setzero_pd();

        for (int iter = 0; iter < NUM_ITERS; ++iter) {
            __m256d r_iter = _mm256_add_pd(r_vec, _mm256_set1_pd(1e-4 * iter));
            __m256d r_plus_hvs = _mm256_add_pd(r_iter, half_v_squared);
            __m256d num = _mm256_fmadd_pd(r_plus_hvs, t_vec, log_sk_vec);
            __m256d d1_vec = _mm256_div_pd(num, v_sqrt_t);
            __m256d d2_vec = _mm256_sub_pd(d1_vec, v_sqrt_t);

            __m256d cnd_d1 = cnd_vec(d1_vec);
            __m256d cnd_m_d1 = _mm256_sub_pd(one, cnd_d1);
            __m256d cnd_d2 = cnd_vec(d2_vec);
            __m256d cnd_m_d2 = _mm256_sub_pd(one, cnd_d2);

            __m256d exp_disc_arg = _mm256_mul_pd(_mm256_sub_pd(zero, r_iter), t_vec);
            double exp_disc_arr[4];
            _mm256_storeu_pd(exp_disc_arr, exp_disc_arg);
            double disc_exp_arr[4];
            for (int l = 0; l < 4; ++l) disc_exp_arr[l] = std::exp(exp_disc_arr[l]);
            __m256d disc_exp_vec = _mm256_loadu_pd(disc_exp_arr);
            __m256d disc_vec = _mm256_mul_pd(k_vec, disc_exp_vec);

            __m256d term1 = _mm256_mul_pd(s_vec, cnd_d1);
            __m256d term2 = _mm256_mul_pd(disc_vec, cnd_d2);
            __m256d call_vec = _mm256_sub_pd(term1, term2);

            __m256d term3 = _mm256_mul_pd(disc_vec, cnd_m_d2);
            __m256d term4 = _mm256_mul_pd(s_vec, cnd_m_d1);
            __m256d put_vec = _mm256_sub_pd(term3, term4);

            call_acc_vec = _mm256_add_pd(call_acc_vec, call_vec);
            put_acc_vec  = _mm256_add_pd(put_acc_vec, put_vec);
        }

        _mm256_store_pd(&call_acc[i], call_acc_vec);
        _mm256_store_pd(&put_acc[i],  put_acc_vec);
    }

    // scalar tail for this rank
    for (int i = local_end - (local_n % vec_width); i < local_end; ++i) {
        double s_val = S[i];
        double k_val = K[i];
        double t_val = T[i];
        double r_val = R[i];
        double v_val = V[i];

        double log_sk = std::log(s_val / k_val);
        double v_squared = v_val * v_val;
        double half_v_squared = 0.5 * v_squared;
        double sqrt_t = std::sqrt(t_val);
        double v_sqrt_t = v_val * sqrt_t;

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
        put_acc[i] = put_accum;
    }

    double end = MPI_Wtime();
    double elapsed_ms = (end - start) * 1000.0;

    double max_elapsed_ms = 0.0;
    MPI_Reduce(&elapsed_ms, &max_elapsed_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    double local_checksum = 0.0;
    for (int i = local_start; i < local_end; ++i) {
        local_checksum += call_acc[i] + put_acc[i];
    }
    double checksum = 0.0;
    MPI_Reduce(&local_checksum, &checksum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Tiempo de ejecucion de Black-Scholes (%d iters): %.2f ms\n", NUM_ITERS, max_elapsed_ms);
        printf("Checksum: %.6f\n", checksum);
    }

    free(S);
    free(K);
    free(T);
    free(R);
    free(V);
    free(call_acc);
    free(put_acc);

    MPI_Finalize();
    return 0;
}