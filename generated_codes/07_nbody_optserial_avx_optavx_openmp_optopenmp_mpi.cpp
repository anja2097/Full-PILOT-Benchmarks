#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <immintrin.h>
#include <cstddef>
#include <cstdint>
#include <omp.h>
#include <mpi.h>

struct Body {
    double x, y, z;
    double vx, vy, vz;
    double mass;
};

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int N = 16384;
    const int NUM_STEPS = 10;
    const double DT = 0.01;
    const double SOFTENING2 = 1e-4;

    // Compute local chunk indices
    int base = N / size;
    int rem = N % size;
    int local_start = rank * base + (rank < rem ? rank : rem);
    int local_end = local_start + base + (rank < rem ? 1 : 0);

    std::vector<Body> bodies(N);
    std::vector<double> ax(N), ay(N), az(N);

    double *px = (double*)_mm_malloc((size_t)N * sizeof(double), 32);
    double *py = (double*)_mm_malloc((size_t)N * sizeof(double), 32);
    double *pz = (double*)_mm_malloc((size_t)N * sizeof(double), 32);
    double *pm = (double*)_mm_malloc((size_t)N * sizeof(double), 32);

    // NUMA first-touch with same static mapping as timed kernel, before rand data.
    #pragma omp parallel for schedule(static) default(none) shared(px,py,pz,pm,ax,ay,az,bodies)
    for (int i = 0; i < N; ++i) {
        px[i] = 0.0;
        py[i] = 0.0;
        pz[i] = 0.0;
        pm[i] = 0.0;
        ax[i] = 0.0;
        ay[i] = 0.0;
        az[i] = 0.0;
        bodies[i].x = 0.0;
        bodies[i].y = 0.0;
        bodies[i].z = 0.0;
        bodies[i].vx = 0.0;
        bodies[i].vy = 0.0;
        bodies[i].vz = 0.0;
        bodies[i].mass = 0.0;
    }

    srand(42);
    for (int i = 0; i < N; ++i) {
        bodies[i].x = static_cast<double>(rand()) / RAND_MAX;
        bodies[i].y = static_cast<double>(rand()) / RAND_MAX;
        bodies[i].z = static_cast<double>(rand()) / RAND_MAX;
        bodies[i].vx = bodies[i].vy = bodies[i].vz = 0.0;
        bodies[i].mass = (0.5 + static_cast<double>(rand()) / RAND_MAX) / N;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    const __m256d v_soft = _mm256_set1_pd(SOFTENING2);
    const __m256d v_one  = _mm256_set1_pd(1.0);
    const int Nvec = N & ~3;

    double* __restrict__ pxr = px;
    double* __restrict__ pyr = py;
    double* __restrict__ pzr = pz;
    double* __restrict__ pmr = pm;
    double* __restrict__ axr = ax.data();
    double* __restrict__ ayr = ay.data();
    double* __restrict__ azr = az.data();

    for (int step = 0; step < NUM_STEPS; ++step) {
        // Pack AoS -> SoA contiguous panels (full size)
        #pragma omp parallel for schedule(static) default(none) shared(pxr,pyr,pzr,pmr,bodies)
        for (int i = 0; i < N; ++i) {
            pxr[i] = bodies[i].x;
            pyr[i] = bodies[i].y;
            pzr[i] = bodies[i].z;
            pmr[i] = bodies[i].mass;
        }

        // Zero acceleration arrays before accumulation
        #pragma omp parallel for schedule(static) default(none) shared(axr,ayr,azr)
        for (int i = 0; i < N; ++i) {
            axr[i] = 0.0;
            ayr[i] = 0.0;
            azr[i] = 0.0;
        }

        const double* __restrict__ rpx = pxr;
        const double* __restrict__ rpy = pyr;
        const double* __restrict__ rpz = pzr;
        const double* __restrict__ rpm = pmr;
        double* __restrict__ rax = axr;
        double* __restrict__ ray = ayr;
        double* __restrict__ raz = azr;

        // Compute forces for the local chunk
        #pragma omp parallel default(none) shared(rpx,rpy,rpz,rpm,rax,ray,raz,local_start,local_end)
        {
            #pragma omp for schedule(static)
            for (int i0 = local_start; i0 < local_end; i0 += 2) {
                if (i0 + 1 < local_end) {
                    __m256d fx0 = _mm256_setzero_pd();
                    __m256d fy0 = _mm256_setzero_pd();
                    __m256d fz0 = _mm256_setzero_pd();
                    __m256d fx1 = _mm256_setzero_pd();
                    __m256d fy1 = _mm256_setzero_pd();
                    __m256d fz1 = _mm256_setzero_pd();

                    __m256d vxi0 = _mm256_set1_pd(rpx[i0]);
                    __m256d vyi0 = _mm256_set1_pd(rpy[i0]);
                    __m256d vzi0 = _mm256_set1_pd(rpz[i0]);
                    __m256d vxi1 = _mm256_set1_pd(rpx[i0 + 1]);
                    __m256d vyi1 = _mm256_set1_pd(rpy[i0 + 1]);
                    __m256d vzi1 = _mm256_set1_pd(rpz[i0 + 1]);

                    for (int j = 0; j < Nvec; j += 4) {
                        __m256d xj = _mm256_load_pd(rpx + j);
                        __m256d yj = _mm256_load_pd(rpy + j);
                        __m256d zj = _mm256_load_pd(rpz + j);
                        __m256d mj = _mm256_load_pd(rpm + j);

                        __m256d dx0 = _mm256_sub_pd(xj, vxi0);
                        __m256d dy0 = _mm256_sub_pd(yj, vyi0);
                        __m256d dz0 = _mm256_sub_pd(zj, vzi0);
                        __m256d r20 = _mm256_mul_pd(dx0, dx0);
                        r20 = _mm256_fmadd_pd(dy0, dy0, r20);
                        r20 = _mm256_fmadd_pd(dz0, dz0, r20);
                        r20 = _mm256_add_pd(r20, v_soft);
                        __m256d inv0 = _mm256_div_pd(v_one, _mm256_sqrt_pd(r20));
                        __m256d inv20 = _mm256_mul_pd(inv0, inv0);
                        __m256d inv30 = _mm256_mul_pd(inv0, inv20);
                        __m256d s0 = _mm256_mul_pd(mj, inv30);
                        fx0 = _mm256_fmadd_pd(s0, dx0, fx0);
                        fy0 = _mm256_fmadd_pd(s0, dy0, fy0);
                        fz0 = _mm256_fmadd_pd(s0, dz0, fz0);

                        __m256d dx1 = _mm256_sub_pd(xj, vxi1);
                        __m256d dy1 = _mm256_sub_pd(yj, vyi1);
                        __m256d dz1 = _mm256_sub_pd(zj, vzi1);
                        __m256d r21 = _mm256_mul_pd(dx1, dx1);
                        r21 = _mm256_fmadd_pd(dy1, dy1, r21);
                        r21 = _mm256_fmadd_pd(dz1, dz1, r21);
                        r21 = _mm256_add_pd(r21, v_soft);
                        __m256d inv1 = _mm256_div_pd(v_one, _mm256_sqrt_pd(r21));
                        __m256d inv21 = _mm256_mul_pd(inv1, inv1);
                        __m256d inv31 = _mm256_mul_pd(inv1, inv21);
                        __m256d s1 = _mm256_mul_pd(mj, inv31);
                        fx1 = _mm256_fmadd_pd(s1, dx1, fx1);
                        fy1 = _mm256_fmadd_pd(s1, dy1, fy1);
                        fz1 = _mm256_fmadd_pd(s1, dz1, fz1);
                    }

                    alignas(32) double tmp[4];
                    double fx_tail0 = 0.0, fy_tail0 = 0.0, fz_tail0 = 0.0;
                    double fx_tail1 = 0.0, fy_tail1 = 0.0, fz_tail1 = 0.0;
                    for (int j = Nvec; j < N; ++j) {
                        {
                            double dx = rpx[j] - rpx[i0];
                            double dy = rpy[j] - rpy[i0];
                            double dz = rpz[j] - rpz[i0];
                            double r2 = dx * dx + dy * dy + dz * dz + SOFTENING2;
                            double inv_r = 1.0 / std::sqrt(r2);
                            double inv_r3 = inv_r * inv_r * inv_r;
                            double s = rpm[j] * inv_r3;
                            fx_tail0 += s * dx;
                            fy_tail0 += s * dy;
                            fz_tail0 += s * dz;
                        }
                        {
                            double dx = rpx[j] - rpx[i0 + 1];
                            double dy = rpy[j] - rpy[i0 + 1];
                            double dz = rpz[j] - rpz[i0 + 1];
                            double r2 = dx * dx + dy * dy + dz * dz + SOFTENING2;
                            double inv_r = 1.0 / std::sqrt(r2);
                            double inv_r3 = inv_r * inv_r * inv_r;
                            double s = rpm[j] * inv_r3;
                            fx_tail1 += s * dx;
                            fy_tail1 += s * dy;
                            fz_tail1 += s * dz;
                        }
                    }

                    _mm256_store_pd(tmp, fx0);
                    double fx0s = tmp[0] + tmp[1] + tmp[2] + tmp[3];
                    _mm256_store_pd(tmp, fy0);
                    double fy0s = tmp[0] + tmp[1] + tmp[2] + tmp[3];
                    _mm256_store_pd(tmp, fz0);
                    double fz0s = tmp[0] + tmp[1] + tmp[2] + tmp[3];
                    _mm256_store_pd(tmp, fx1);
                    double fx1s = tmp[0] + tmp[1] + tmp[2] + tmp[3];
                    _mm256_store_pd(tmp, fy1);
                    double fy1s = tmp[0] + tmp[1] + tmp[2] + tmp[3];
                    _mm256_store_pd(tmp, fz1);
                    double fz1s = tmp[0] + tmp[1] + tmp[2] + tmp[3];

                    rax[i0] = fx0s + fx_tail0;
                    ray[i0] = fy0s + fy_tail0;
                    raz[i0] = fz0s + fz_tail0;
                    rax[i0 + 1] = fx1s + fx_tail1;
                    ray[i0 + 1] = fy1s + fy_tail1;
                    raz[i0 + 1] = fz1s + fz_tail1;
                } else {
                    __m256d fx_vec = _mm256_setzero_pd();
                    __m256d fy_vec = _mm256_setzero_pd();
                    __m256d fz_vec = _mm256_setzero_pd();
                    __m256d vxi = _mm256_set1_pd(rpx[i0]);
                    __m256d vyi = _mm256_set1_pd(rpy[i0]);
                    __m256d vzi = _mm256_set1_pd(rpz[i0]);
                    for (int j = 0; j < Nvec; j += 4) {
                        __m256d xj = _mm256_load_pd(rpx + j);
                        __m256d yj = _mm256_load_pd(rpy + j);
                        __m256d zj = _mm256_load_pd(rpz + j);
                        __m256d mj = _mm256_load_pd(rpm + j);
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
                    double fx_tail = 0.0, fy_tail = 0.0, fz_tail = 0.0;
                    for (int j = Nvec; j < N; ++j) {
                        double dx = rpx[j] - rpx[i0];
                        double dy = rpy[j] - rpy[i0];
                        double dz = rpz[j] - rpz[i0];
                        double r2 = dx * dx + dy * dy + dz * dz + SOFTENING2;
                        double inv_r = 1.0 / std::sqrt(r2);
                        double inv_r3 = inv_r * inv_r * inv_r;
                        double s = rpm[j] * inv_r3;
                        fx_tail += s * dx;
                        fy_tail += s * dy;
                        fz_tail += s * dz;
                    }
                    alignas(32) double tmp[4];
                    _mm256_store_pd(tmp, fx_vec);
                    double fxs = tmp[0] + tmp[1] + tmp[2] + tmp[3];
                    _mm256_store_pd(tmp, fy_vec);
                    double fys = tmp[0] + tmp[1] + tmp[2] + tmp[3];
                    _mm256_store_pd(tmp, fz_vec);
                    double fzs = tmp[0] + tmp[1] + tmp[2] + tmp[3];
                    rax[i0] = fxs + fx_tail;
                    ray[i0] = fys + fy_tail;
                    raz[i0] = fzs + fz_tail;
                }
            }
        } // end parallel region for force computation

        // Reduce accelerations across ranks
        MPI_Allreduce(MPI_IN_PLACE, axr, N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, ayr, N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, azr, N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        // Update velocities and positions (full domain)
        #pragma omp parallel for schedule(static) default(none) shared(bodies,axr,ayr,azr)
        for (int i = 0; i < N; ++i) {
            bodies[i].vx += axr[i] * DT;
            bodies[i].vy += ayr[i] * DT;
            bodies[i].vz += azr[i] * DT;
            bodies[i].x  += bodies[i].vx * DT;
            bodies[i].y  += bodies[i].vy * DT;
            bodies[i].z  += bodies[i].vz * DT;
        }
    }

    double end = MPI_Wtime();
    double elapsed_ms = (end - start) * 1000.0;

    double max_elapsed_ms = 0.0;
    MPI_Reduce(&elapsed_ms, &max_elapsed_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Tiempo de ejecucion del N-body (%d pasos): %.2f ms\n", NUM_STEPS, max_elapsed_ms);
    }

    double checksum = 0.0;
    for (int i = 0; i < N; ++i) {
        double speed = std::sqrt(bodies[i].vx * bodies[i].vx +
                                 bodies[i].vy * bodies[i].vy +
                                 bodies[i].vz * bodies[i].vz);
        checksum += speed * static_cast<double>(i % 5 + 1);
    }
    if (rank == 0) {
        printf("Checksum: %.6f\n", checksum);
    }

    _mm_free(px);
    _mm_free(py);
    _mm_free(pz);
    _mm_free(pm);

    MPI_Finalize();
    return 0;
}