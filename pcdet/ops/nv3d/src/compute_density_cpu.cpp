#include <math.h>
#include <stdio.h>
#include <nanoflann.hpp>
#include <vector>
#include <stdlib.h>
#include <torch/serialize/tensor.h>

#include "compute_density_cpu.h"

using namespace nanoflann;
using namespace std;

void compute_density_cpu(at::Tensor pc_tensor,
                         int N,
                         float radius,
                         at::Tensor den_tensor) {
                             
    auto pc_ptr = pc_tensor.data_ptr<float>();
    auto den_ptr = den_tensor.data_ptr<float>();
    
    float r2 = radius * radius; // Squared search radius
    
    float block_min = numeric_limits<float>::max();
    float block_max = -numeric_limits<float>::max();

    // Compute densities
    for (int n = 0; n < N; n++) {
        const float* current_point = pc_ptr + n * 3;
        float rx = current_point[0];
        float ry = current_point[1];
        float rz = current_point[2];

        int count = 0;
        for (int i = 0; i < N; i++) {
            const float* neighbor = pc_ptr + i * 3;
            float dx = neighbor[0] - rx;
            float dy = neighbor[1] - ry;
            float dz = neighbor[2] - rz;
            float dist2 = dx * dx + dy * dy + dz * dz;
            if (dist2 <= r2) {
                count++;
            }
        }
        den_ptr[n] = static_cast<float>(count);

        // Update min and max
        block_min = min(block_min, den_ptr[n]);
        block_max = max(block_max, den_ptr[n]);
    }

    // Normalize densities
    float denom = block_max - block_min + 1e-6f;
    for (int n = 0; n < N; n++) {
        den_ptr[n] = (den_ptr[n] - block_min) / denom;
    }
}
