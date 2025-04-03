/*
batch version of seven-nn, modified from the original implementation of Shaoshuai Shi codes.
Written by Krittin Chaowakarn
All Rights Reserved 2025.
*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <curand_kernel.h>

#include "cuda_utils.h"
#include "compute_norm_mask_gpu.h"


__global__ void compute_norm_mask_kernel_fast(const int *density, 
                                              float *normalized_density,
                                              int B, int N, float drop_rate, float threshold,
                                              const int *min_vals, const int *max_vals,
                                              const float *random_vals,
                                              const bool *mask_in, bool *mask_out) {
    int bs_idx = blockIdx.x;  // Batch index.
    int pt_idx = threadIdx.x + blockIdx.y * blockDim.x;
    int global_idx = bs_idx * N + pt_idx;
    
    if (bs_idx >= B || pt_idx >= N || !mask_in[global_idx]) return;

    // Load min and max once per batch using Shared Memory (if batch size is small)
    __shared__ int shared_min, shared_max;
    if (threadIdx.x == 0) {
        shared_min = min_vals[bs_idx];
        shared_max = max_vals[bs_idx];
    }
    __syncthreads();  // Ensure shared memory is loaded

    // Read shared memory instead of global memory
    int global_min = shared_min;
    int global_max = shared_max;

    int local_density = density[global_idx];

    // Avoid division in each thread: precompute the inverse
    float range_inv = 1.0f / (global_max - global_min + 1e-8f);
    float norm_density = (local_density - global_min) * range_inv;
    normalized_density[global_idx] = norm_density;

    // Use && for short-circuit evaluation
    if (norm_density > threshold && random_vals[global_idx] < drop_rate) {
        mask_out[global_idx] = false;
    }
}

void compute_norm_mask_kernel_launcher_fast(const int *density, 
                                                       float *normalized_density, 
                                                       int B, int N, float drop_rate, float threshold,
                                                       const int *min_vals, const int *max_vals,
                                                       const float *random_vals,
                                                       const bool *mask_in, bool *mask_out) {
    dim3 grid(B, DIVUP(N, THREADS_PER_BLOCK));
    dim3 block(THREADS_PER_BLOCK);
    compute_norm_mask_kernel_fast<<<grid, block>>>(density, normalized_density,
                                                     B, N, drop_rate, threshold,
                                                     min_vals, max_vals,
                                                     random_vals, mask_in, mask_out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}


