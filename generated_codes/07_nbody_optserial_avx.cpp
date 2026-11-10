/* ANALYSIS: N-body O(N^2) gravity, compute‑bound (sqrt, mul, add). Vectorized inner j‑loop with AVX2 256‑bit doubles, gather loads for body fields, SIMD sqrt & FMAs, horizontal reduction to scalars. */
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <immintrin.h>
#include <cstddef>
#include <cstdint>

struct Body {
    double x, y, z;
    double vx, vy, vz;
    double mass;
};

int main()
{
    const int N = 16384;
    const int NUM_STEPS = 10;
    const double DT = 0.01;
    const double SOFTENING2 = 1e-4;

    std::vector<Body> bodies(N);
    std::vector<double> ax(N), ay(N), az(N);

    srand(42);
    for (int i = 0; i < N; ++i) {
        bodies[i].x = static_cast<double>(rand()) / RAND_MAX;
        bodies[i].y = static_cast<double>(rand()) / RAND_MAX;
        bodies[i].z = static_cast<double>(rand()) / RAND_MAX;
        bodies[i].vx = bodies[i].vy = bodies[i].vz = 0.0;
        bodies[i].mass = (0.5 + static_cast<double>(rand()) / RAND_MAX) / N;
    }

    auto start = std::chrono::high_resolution_clock::now();

    const size_t stride = sizeof(Body);
    const double* base_ptr = reinterpret_cast<const double*>(bodies.data());
    const __m256d v_soft = _mm256_set1_pd(SOFTENING2);
    const __m256d v_one  = _mm256_set1_pd(1.0);

    for (int step = 0; step < NUM_STEPS; ++step) {
        for (int i = 0; i < N; ++i) {
            __m256d fx_vec = _mm256_setzero_pd();
            __m256d fy_vec = _mm256_setzero_pd();
            __m256d fz_vec = _mm256_setzero_pd();

            __m256d vxi = _mm256_set1_pd(bodies[i].x);
            __m256d vyi = _mm256_set1_pd(bodies[i].y);
            __m256d vzi = _mm256_set1_pd(bodies[i].z);

            int j = 0;
            for (; j <= N - 4; j += 4) {
                // gather x
                __m256i idx_x = _mm256_set_epi64x(
                    (int64_t)((j + 3) * stride + offsetof(Body, x)),
                    (int64_t)((j + 2) * stride + offsetof(Body, x)),
                    (int64_t)((j + 1) * stride + offsetof(Body, x)),
                    (int64_t)((j + 0) * stride + offsetof(Body, x)));
                __m256d xj = _mm256_i64gather_pd(base_ptr, idx_x, 1);

                // gather y
                __m256i idx_y = _mm256_set_epi64x(
                    (int64_t)((j + 3) * stride + offsetof(Body, y)),
                    (int64_t)((j + 2) * stride + offsetof(Body, y)),
                    (int64_t)((j + 1) * stride + offsetof(Body, y)),
                    (int64_t)((j + 0) * stride + offsetof(Body, y)));
                __m256d yj = _mm256_i64gather_pd(base_ptr, idx_y, 1);

                // gather z
                __m256i idx_z = _mm256_set_epi64x(
                    (int64_t)((j + 3) * stride + offsetof(Body, z)),
                    (int64_t)((j + 2) * stride + offsetof(Body, z)),
                    (int64_t)((j + 1) * stride + offsetof(Body, z)),
                    (int64_t)((j + 0) * stride + offsetof(Body, z)));
                __m256d zj = _mm256_i64gather_pd(base_ptr, idx_z, 1);

                // gather mass
                __m256i idx_m = _mm256_set_epi64x(
                    (int64_t)((j + 3) * stride + offsetof(Body, mass)),
                    (int64_t)((j + 2) * stride + offsetof(Body, mass)),
                    (int64_t)((j + 1) * stride + offsetof(Body, mass)),
                    (int64_t)((j + 0) * stride + offsetof(Body, mass)));
                __m256d mj = _mm256_i64gather_pd(base_ptr, idx_m, 1);

                __m256d dx = _mm256_sub_pd(xj, vxi);
                __m256d dy = _mm256_sub_pd(yj, vyi);
                __m256d dz = _mm256_sub_pd(zj, vzi);

                __m256d r2 = _mm256_mul_pd(dx, dx);
                r2 = _mm256_fmadd_pd(dy, dy, r2);
                r2 = _mm256_fmadd_pd(dz, dz, r2);
                r2 = _mm256_add_pd(r2, v_soft);

                __m256d inv_r = _mm256_div_pd(v_one, _mm256_sqrt_pd(r2));
                __m256d inv_r2 = _mm256_mul_pd(inv_r, inv_r);
                __m256d inv_r3 = _mm256_mul_pd(inv_r, inv_r2);

                __m256d s = _mm256_mul_pd(mj, inv_r3);
                fx_vec = _mm256_fmadd_pd(s, dx, fx_vec);
                fy_vec = _mm256_fmadd_pd(s, dy, fy_vec);
                fz_vec = _mm256_fmadd_pd(s, dz, fz_vec);
            }

            // Tail scalar loop
            double fx_tail = 0.0, fy_tail = 0.0, fz_tail = 0.0;
            for (; j < N; ++j) {
                double dx = bodies[j].x - bodies[i].x;
                double dy = bodies[j].y - bodies[i].y;
                double dz = bodies[j].z - bodies[i].z;
                double r2 = dx * dx + dy * dy + dz * dz + SOFTENING2;
                double inv_r = 1.0 / std::sqrt(r2);
                double inv_r3 = inv_r * inv_r * inv_r;
                double s = bodies[j].mass * inv_r3;
                fx_tail += s * dx;
                fy_tail += s * dy;
                fz_tail += s * dz;
            }

            // Horizontal reduction of vector lanes
            double fx_vec_sum = 0.0, fy_vec_sum = 0.0, fz_vec_sum = 0.0;
            alignas(32) double tmp[4];
            _mm256_store_pd(tmp, fx_vec);
            fx_vec_sum = tmp[0] + tmp[1] + tmp[2] + tmp[3];
            _mm256_store_pd(tmp, fy_vec);
            fy_vec_sum = tmp[0] + tmp[1] + tmp[2] + tmp[3];
            _mm256_store_pd(tmp, fz_vec);
            fz_vec_sum = tmp[0] + tmp[1] + tmp[2] + tmp[3];

            ax[i] = fx_vec_sum + fx_tail;
            ay[i] = fy_vec_sum + fy_tail;
            az[i] = fz_vec_sum + fz_tail;
        }

        for (int i = 0; i < N; ++i) {
            bodies[i].vx += ax[i] * DT;
            bodies[i].vy += ay[i] * DT;
            bodies[i].vz += az[i] * DT;
            bodies[i].x  += bodies[i].vx * DT;
            bodies[i].y  += bodies[i].vy * DT;
            bodies[i].z  += bodies[i].vz * DT;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    printf("Tiempo de ejecucion del N-body (%d pasos): %.2f ms\n", NUM_STEPS, elapsed.count());

    double checksum = 0.0;
    for (int i = 0; i < N; ++i) {
        double speed = std::sqrt(bodies[i].vx * bodies[i].vx +
                                 bodies[i].vy * bodies[i].vy +
                                 bodies[i].vz * bodies[i].vz);
        checksum += speed * static_cast<double>(i % 5 + 1);
    }
    printf("Checksum: %.6f\n", checksum);
    return 0;
}