/* ANALYSIS: 
 * Kernel type: Black-Scholes option pricing, compute-bound but limited by memory bandwidth due to large arrays.
 * Likely bottleneck: Memory bandwidth (working set per option per iteration is large, but we reduce by swapping loops).
 * Transformations applied: 
 *   1. Loop interchange: swapped the iteration and option loops to reuse input data across iterations.
 *   2. Loop-invariant code motion: moved computations that are constant per option out of the iteration loop.
 *   3. Memory allocation: switched to aligned_alloc for 64-byte alignment to support future vectorization.
 *   4. Added __restrict__ to pointers to improve compiler alias analysis.
 * Why: Reduces memory traffic from 7*N*NUM_ITERS*sizeof(double) to 7*N*sizeof(double) by reusing input data across iterations.
 *        Also reduces computational overhead by moving invariant operations out of the inner loop.
 */
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

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

    // Swapped loops: outer over options, inner over iterations
    for (int i = 0; i < N; ++i) {
        // Restrict pointers to help compiler alias analysis
        double* __restrict__ s_ptr = S;
        double* __restrict__ k_ptr = K;
        double* __restrict__ t_ptr = T;
        double* __restrict__ r_ptr = R;
        double* __restrict__ v_ptr = V;
        double* __restrict__ call_acc_ptr = call_acc;
        double* __restrict__ put_acc_ptr = put_acc;

        double s_val = s_ptr[i];
        double k_val = k_ptr[i];
        double t_val = t_ptr[i];
        double r_val = r_ptr[i];
        double v_val = v_ptr[i];

        // Precompute invariant terms per option
        double log_sk = std::log(s_val / k_val);
        double v_squared = v_val * v_val;
        double half_v_squared = 0.5 * v_squared;
        double sqrt_t = std::sqrt(t_val);
        double v_sqrt_t = v_val * sqrt_t; // denominator for d1

        double call_accum = 0.0;
        double put_accum = 0.0;
        for (int iter = 0; iter < NUM_ITERS; ++iter) {
            double r = r_val + 1e-4 * iter;
            double d1 = (log_sk + (r + half_v_squared) * t_val) / v_sqrt_t;
            double d2 = d1 - v_sqrt_t;
            double disc = k_val * std::exp(-r * t_val);
            double call = s_val * cnd(d1) - disc * cnd(d2);
            double put = disc * cnd(-d2) - s_val * cnd(-d1);
            call_accum += call;
            put_accum += put;
        }

        call_acc_ptr[i] = call_accum;
        put_acc_ptr[i] = put_accum;
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