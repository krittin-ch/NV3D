#include <math.h>
#include <stdio.h>
#include <vector>
#include <stdlib.h>
#include <torch/serialize/tensor.h>
#include <nanoflann.hpp>
#include <cfloat>

#include "compute_seven_nn_cpu.h"

using namespace nanoflann;
using namespace std;

void compute_seven_nn_cpu(int b, int n, int m, 
                  at::Tensor& unknown_tensor,  // (B, N, 3)
                  at::Tensor& known_tensor,    // (B, M, 3)
                  at::Tensor& dist2_tensor,          // Output: (B, N, 7)
                  at::Tensor& idx_tensor) {          // Output: (B, N, 7)
    
    // Ensure correct tensor types
    TORCH_CHECK(unknown_tensor.dtype() == torch::kFloat32, "unknown_tensor must be float32");
    TORCH_CHECK(known_tensor.dtype() == torch::kFloat32, "known_tensor must be float32");
    TORCH_CHECK(dist2_tensor.dtype() == torch::kFloat32, "dist2_tensor must be float32");
    TORCH_CHECK(idx_tensor.dtype() == torch::kInt64, "idx_tensor must be int64");

    // Get pointers to raw tensor data
    const float* unknown = unknown_tensor.data_ptr<float>();   // (B, N, 3)
    const float* known = known_tensor.data_ptr<float>();       // (B, M, 3)
    float* dist2 = dist2_tensor.data_ptr<float>();             // (B, N, 7)
    int64_t* idx = idx_tensor.data_ptr<int64_t>();                 // (B, N, 7)

    // Loop over batches
    for (int bs_idx = 0; bs_idx < b; ++bs_idx) {
        for (int pt_idx = 0; pt_idx < n; ++pt_idx) {
            // Pointers to current unknown point
            const float* unknown_point = unknown + bs_idx * n * 3 + pt_idx * 3;
            float ux = unknown_point[0];
            float uy = unknown_point[1];
            float uz = unknown_point[2];

            // Initialize best distances and indices
            float best_d[7] = {FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX};
            int64_t best_idx[7] = {-1, -1, -1, -1, -1, -1, -1};

            // Search over all known points
            for (int k = 0; k < m; ++k) {
                const float* known_point = known + bs_idx * m * 3 + k * 3;
                float x = known_point[0];
                float y = known_point[1];
                float z = known_point[2];

                // Compute squared Euclidean distance
                float d = (ux - x) * (ux - x) + (uy - y) * (uy - y) + (uz - z) * (uz - z);

                // If this is closer than the worst of the top 7, insert it
                if (d < best_d[6]) {
                    best_d[6] = d;
                    best_idx[6] = k;

                    // Sort the distances in ascending order (Bubble Insert)
                    for (int i = 5; i >= 0; --i) {
                        if (best_d[i + 1] < best_d[i]) {
                            // Swap distances
                            swap(best_d[i], best_d[i + 1]);
                            // Swap indices
                            swap(best_idx[i], best_idx[i + 1]);
                        } else {
                            break;
                        }
                    }
                }
            }

            // Store results in output tensors
            for (int i = 0; i < 7; ++i) {
                dist2[bs_idx * n * 7 + pt_idx * 7 + i] = best_d[i];
                idx[bs_idx * n * 7 + pt_idx * 7 + i] = best_idx[i];
            }
        }
    }
}