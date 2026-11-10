// N-body gravitacional con suma directa O(N^2) e integracion de Euler simplectica.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

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

    for (int step = 0; step < NUM_STEPS; ++step) {
        for (int i = 0; i < N; ++i) {
            double fx = 0.0, fy = 0.0, fz = 0.0;
            for (int j = 0; j < N; ++j) {
                double dx = bodies[j].x - bodies[i].x;
                double dy = bodies[j].y - bodies[i].y;
                double dz = bodies[j].z - bodies[i].z;
                double r2 = dx * dx + dy * dy + dz * dz + SOFTENING2;
                double inv_r = 1.0 / std::sqrt(r2);
                double inv_r3 = inv_r * inv_r * inv_r;
                fx += bodies[j].mass * dx * inv_r3;
                fy += bodies[j].mass * dy * inv_r3;
                fz += bodies[j].mass * dz * inv_r3;
            }
            ax[i] = fx;
            ay[i] = fy;
            az[i] = fz;
        }
        for (int i = 0; i < N; ++i) {
            bodies[i].vx += ax[i] * DT;
            bodies[i].vy += ay[i] * DT;
            bodies[i].vz += az[i] * DT;
            bodies[i].x += bodies[i].vx * DT;
            bodies[i].y += bodies[i].vy * DT;
            bodies[i].z += bodies[i].vz * DT;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    printf("Tiempo de ejecucion del N-body (%d pasos): %.2f ms\n", NUM_STEPS, elapsed.count());

    double checksum = 0.0;
    for (int i = 0; i < N; ++i) {
        double speed = std::sqrt(bodies[i].vx * bodies[i].vx + bodies[i].vy * bodies[i].vy +
                                 bodies[i].vz * bodies[i].vz);
        checksum += speed * static_cast<double>(i % 5 + 1);
    }
    printf("Checksum: %.6f\n", checksum);
    return 0;
}
