/*
batch version of seven-nn, modified from the original implementation of Shaoshuai Shi codes.
Written by Krittin Chaowakarn
All Rights Reserved 2025.
*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "cuda_utils.h"
#include "compute_seven_nn_gpu.h"

#define K 7


/*
__device__ void insert_sorted(float best_d[K], int64_t best_idx[K], float new_dist, int64_t new_idx) {
    if (new_dist >= best_d[K - 1]) return;  // Skip if farther than worst stored neighbor

    int i = K - 1;
    while (i > 0 && best_d[i - 1] > new_dist) {
        best_d[i] = best_d[i - 1];
        best_idx[i] = best_idx[i - 1];
        i--;
    }
    best_d[i] = new_dist;
    best_idx[i] = new_idx;
}

__global__ void compute_seven_nn_kernel_fast(int b, int nq, int nd, 
    const float *__restrict__ query_points, 
    const float *__restrict__ database_points, 
    float *__restrict__ dist2, 
    int64_t *__restrict__ idx,
    const bool *__restrict__ query_mask,
    const bool *__restrict__ db_mask) {

    int bs_idx = blockIdx.y;  // Batch index
    int q_idx = blockIdx.x * blockDim.x + threadIdx.x;  // Query index in batch

    if (bs_idx >= b || q_idx >= nq || !query_mask[bs_idx * nq + q_idx]) return;

    float qx = query_points[bs_idx * nq * 3 + q_idx * 3 + 0];
    float qy = query_points[bs_idx * nq * 3 + q_idx * 3 + 1];
    float qz = query_points[bs_idx * nq * 3 + q_idx * 3 + 2];

    // Best 7 distances and indices
    float best_d[K] = {FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX};
    int64_t best_idx[K] = {-1, -1, -1, -1, -1, -1, -1};

    // Shared memory for database points (Each thread block loads part of the database)
    __shared__ float shared_db[SHARED_MEM_SIZE][3];  
    __shared__ bool shared_mask[SHARED_MEM_SIZE];

    for (int block_offset = 0; block_offset < nd; block_offset += SHARED_MEM_SIZE) {
        int db_idx = block_offset + threadIdx.x;

        if (db_idx < nd) {
            shared_db[threadIdx.x][0] = database_points[bs_idx * nd * 3 + db_idx * 3 + 0];
            shared_db[threadIdx.x][1] = database_points[bs_idx * nd * 3 + db_idx * 3 + 1];
            shared_db[threadIdx.x][2] = database_points[bs_idx * nd * 3 + db_idx * 3 + 2];
            shared_mask[threadIdx.x] = db_mask[bs_idx * nd + db_idx];
        } else {
            shared_mask[threadIdx.x] = false;
        }
        __syncthreads();

        // Each query thread compares against all loaded database points
        for (int j = 0; j < SHARED_MEM_SIZE && (block_offset + j) < nd; ++j) {
            if (!shared_mask[j]) continue;

            float dx = shared_db[j][0];
            float dy = shared_db[j][1];
            float dz = shared_db[j][2];

            float d = (qx - dx) * (qx - dx) + (qy - dy) * (qy - dy) + (qz - dz) * (qz - dz);
            
            insert_sorted(best_d, best_idx, d, block_offset + j);
        }
        __syncthreads();  // Ensure all threads finish before loading next chunk
    }

    // Store results
    for (int i = 0; i < K; ++i) {
        dist2[bs_idx * nq * K + q_idx * K + i] = best_d[i];
        idx[bs_idx * nq * K + q_idx * K + i] = best_idx[i];
    }
}

void compute_seven_nn_kernel_launcher_fast(int b, int n, 
    const float *db_points, 
    float *dist2, int64_t *idx, 
    const bool *db_mask) {

    cudaError_t err;
    
    // Allocate memory for query points on GPU
    float *dq_points;
    cudaMalloc((void**)&dq_points, b * n * 3 * sizeof(float));
    
    bool *dq_mask;
    cudaMalloc((void**)&dq_mask, b * n * sizeof(bool));
    
    // Copy database points to query points
    cudaMemcpy(dq_points, db_points, b * n * 3 * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(dq_mask, db_mask, b * n * sizeof(bool), cudaMemcpyDeviceToDevice);

    // Set up CUDA grid and block sizes
    dim3 blocks((n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, b);
    dim3 threads(THREADS_PER_BLOCK);

    // Launch CUDA kernel with copied query points
    compute_seven_nn_kernel_fast<<<blocks, threads>>>(b, n, n, dq_points, db_points, dist2, idx, dq_mask, db_mask);

    // Check for CUDA errors
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed: %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    // Free allocated memory
    cudaFree(dq_points);
    cudaFree(dq_mask);
}
*/












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


__global__ void compute_seven_nn_kernel_fast(int b, int n, 
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
    float best_d[K] = {FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX};
    int64_t best_idx[K] = {-1, -1, -1, -1, -1, -1, -1};
    
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

void compute_seven_nn_kernel_launcher_fast(const float *db_points, 
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
    compute_seven_nn_kernel_fast<<<blocks, threads>>>(b, n, dq_points, db_points, dist2, idx, db_mask);

    // Check for CUDA errors
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed: %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    // Free allocated memory
    cudaFree(dq_points);
}













/*
__device__ void insert_sorted(float best_d[K], int64_t best_idx[K], float new_dist, int64_t new_idx) {
    if (new_dist >= best_d[K - 1]) return;  // Skip if farther than worst stored neighbor

    int i = K - 1;
    while (i > 0 && best_d[i - 1] > new_dist) {
        best_d[i] = best_d[i - 1];
        best_idx[i] = best_idx[i - 1];
        i--;
    }
    best_d[i] = new_dist;
    best_idx[i] = new_idx;
}


__global__ void compute_seven_nn_kernel_fast(int b, int n, 
    const float *__restrict__ query_points, 
    const float *__restrict__ database_points, 
    float *__restrict__ dist2, 
    int64_t *__restrict__ idx,
    const int *bin,
    const bool *__restrict__ db_mask) {

    int bs_idx = blockIdx.y;  // Batch index
    int q_idx = blockIdx.x * blockDim.x + threadIdx.x;  // Query index in batch

    if (bs_idx >= b || q_idx >= n || !db_mask[bs_idx * n + q_idx]) return;

    float qx = query_points[bs_idx * n * 3 + q_idx * 3 + 0];
    float qy = query_points[bs_idx * n * 3 + q_idx * 3 + 1];
    float qz = query_points[bs_idx * n * 3 + q_idx * 3 + 2];
    int q_bin_id = bin[bs_idx * n + q_idx];

    // Best 7 distances and inices
    float best_d[K] = {FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX};
    int64_t best_idx[K] = {-1, -1, -1, -1, -1, -1, -1};
    
    best_d[0] = 0.0;
    best_idx[0] = q_idx;

    // Shared memory for database points (Each thread block loads part of the database)
    __shared__ float shared_db[SHARED_MEM_SIZE][3];  
    __shared__ bool shared_mask[SHARED_MEM_SIZE];
    __shared__ int shared_bin[SHARED_MEM_SIZE];
    
    int count = 0;
    for (int block_offset = 0; block_offset < n; block_offset += SHARED_MEM_SIZE) {
        int db_idx = block_offset + threadIdx.x;

        if (db_idx < n) {
            shared_db[threadIdx.x][0] = database_points[bs_idx * n * 3 + db_idx * 3 + 0];
            shared_db[threadIdx.x][1] = database_points[bs_idx * n * 3 + db_idx * 3 + 1];
            shared_db[threadIdx.x][2] = database_points[bs_idx * n * 3 + db_idx * 3 + 2];
            
            shared_mask[threadIdx.x] = db_mask[bs_idx * n + db_idx];
            shared_bin[threadIdx.x] = bin[bs_idx * n + db_idx];
        } else {
            shared_mask[threadIdx.x] = false;
            shared_bin[threadIdx.x] = -1;
        }
        __syncthreads();

        // Each query thread compares against all loaded database points
        for (int j = 0; j < SHARED_MEM_SIZE && (block_offset + j) < n; ++j) {
            bool bin_mask = (q_bin_id == shared_bin[j]);
            if (!bin_mask) continue;
            if (!shared_mask[j]) continue;
            
            float dx = shared_db[j][0];
            float dy = shared_db[j][1];
            float dz = shared_db[j][2];

            float d = (qx - dx) * (qx - dx) + (qy - dy) * (qy - dy) + (qz - dz) * (qz - dz);
            
            insert_sorted(best_d, best_idx, d, block_offset + j);
            count++;
        }
        __syncthreads();  // Ensure all threads finish before loading next chunk
    }

    int out_idx = bs_idx * n * K + q_idx * K;
    if (count >= K) {
        for (int i = 0; i < K; ++i) {
            dist2[out_idx + i] = best_d[i];
            idx[out_idx + i] = best_idx[i];
        }
    } else {
        for (int i = 0; i < count; ++i) {
            dist2[out_idx + i] = best_d[i];
            idx[out_idx + i] = best_idx[i];
        }
        
        for (int i = count; i < K; ++i) {
            dist2[out_idx + i] = best_d[0];
            idx[out_idx + i] = best_idx[0];
        }
    }
}

void compute_seven_nn_kernel_launcher_fast(const float *db_points, 
    int b, int n, 
    float *dist2, int64_t *idx, 
    const int *bin,
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
    compute_seven_nn_kernel_fast<<<blocks, threads>>>(b, n, dq_points, db_points, dist2, idx, bin, db_mask);

    // Check for CUDA errors
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed: %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    // Free allocated memory
    cudaFree(dq_points);
}
*/




