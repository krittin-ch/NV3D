/*
Batch version of five-NN, modified from the original implementation of Shaoshuai Shi's codes.
Written by Krittin Chaowakarn
All Rights Reserved 2025.
*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "cuda_utils.h"
#include "compute_five_nn_gpu.h"

#define K 5

__device__ void insert_sorted(float best_d[K], int64_t best_idx[K], float new_dist, int64_t new_idx) {
    if (new_dist >= best_d[K - 1]) return;  // Skip if farther than worst stored neighbor

    int i = K;
    while (i > 0 && best_d[i - 1] > new_dist) {
        best_d[i] = best_d[i - 1];
        best_idx[i] = best_idx[i - 1];
        i--;
    }
    best_d[i] = new_dist;
    best_idx[i] = new_idx;
}


__global__ void compute_five_nn_kernel_fast(int b, int n, 
    const float *__restrict__ query_points, 
    const float *__restrict__ database_points, 
    float *__restrict__ dist2, 
    int64_t *__restrict__ idx,
    const bool *__restrict__ db_mask) {

    int bs_idx = blockIdx.y;  // Batch index
    int q_idx = blockIdx.x * blockDim.x + threadIdx.x;  // Query index in batch

    if (bs_idx >= b || q_idx >= n || !db_mask[bs_idx * n + q_idx]) return;

    float qx = query_points[bs_idx * n * 3 + q_idx * 3 + 0];
    float qy = query_points[bs_idx * n * 3 + q_idx * 3 + 1];
    float qz = query_points[bs_idx * n * 3 + q_idx * 3 + 2];
    
    // Best 7 distances and inices
    float best_d[K] = {FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX};
    int64_t best_idx[K] = {-1, -1, -1, -1, -1};
    
    // Shared memory for database points (Each thread block loads part of the database)
    __shared__ float shared_db[SHARED_MEM_SIZE][3];  
    __shared__ bool shared_mask[SHARED_MEM_SIZE];
    
    for (int block_offset = 0; block_offset < n; block_offset += SHARED_MEM_SIZE) {
        int db_idx = block_offset + threadIdx.x;

        if (db_idx < n) {
            shared_db[threadIdx.x][0] = database_points[bs_idx * n * 3 + db_idx * 3 + 0];
            shared_db[threadIdx.x][1] = database_points[bs_idx * n * 3 + db_idx * 3 + 1];
            shared_db[threadIdx.x][2] = database_points[bs_idx * n * 3 + db_idx * 3 + 2];
            shared_mask[threadIdx.x] = db_mask[bs_idx * n + db_idx];
        } else {
            shared_mask[threadIdx.x] = false;
        }

        __syncthreads();

        // Each query thread compares against all loaded database points
        for (int j = 0; j < SHARED_MEM_SIZE && (block_offset + j) < n; ++j) {
            if (!shared_mask[j]) continue;
            
            float dx = shared_db[j][0];
            float dy = shared_db[j][1];
            float dz = shared_db[j][2];

            float d = (qx - dx) * (qx - dx) + (qy - dy) * (qy - dy) + (qz - dz) * (qz - dz);
            
            insert_sorted(best_d, best_idx, d, block_offset + j);
        }
        __syncthreads();  // Ensure all threads finish before loading next chunk
    }
    
    int out_idx = bs_idx * n * K + q_idx * K;
    for (int i = 0; i < K; ++i) {
        dist2[out_idx + i] = best_d[i];
        idx[out_idx + i] = best_idx[i];
    }
    
}

void compute_five_nn_kernel_launcher_fast(const float *db_points, 
    int b, int n, 
    float *dist2, int64_t *idx, 
    const bool *db_mask) {

    cudaError_t err;
    
    // Allocate memory for query points on GPU
    float *dq_points;
    cudaMalloc((void**)&dq_points, b * n * 3 * sizeof(float));
    
    // Copy database points to query points
    cudaMemcpy(dq_points, db_points, b * n * 3 * sizeof(float), cudaMemcpyDeviceToDevice);

    // Set up CUDA grid an block sizes
    dim3 blocks((n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, b);
    dim3 threads(THREADS_PER_BLOCK);

    // Launch CUDA kernel with copied query points
    compute_five_nn_kernel_fast<<<blocks, threads>>>(b, n, dq_points, db_points, dist2, idx, db_mask);

    // Check for CUDA errors
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed: %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    // Free allocated memory
    cudaFree(dq_points);
}


