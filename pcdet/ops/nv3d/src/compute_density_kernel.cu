#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "cuda_utils.h"
#include "compute_density_gpu.h"



/*
__global__ void compute_density_kernel_fast(const float* __restrict__ point_clouds,
                                            int B, int N,
                                            float radius,
                                            int* __restrict__ densities,
                                            const bool* mask) {
    int bs_idx = blockIdx.y;  // Batch index (0 to B-1)
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x; // Point index in batch
    
    if (bs_idx >= B || pt_idx >= N) return;
    if (!mask[bs_idx * N + pt_idx]) return; 
    
    // Move pointer to correct batch
    const float* batch_start = point_clouds + bs_idx * N * 3;
    const float* current_point = batch_start + pt_idx * 3;
    
    float rx = current_point[0];
    float ry = current_point[1];
    float rz = current_point[2];
    float r2 = radius * radius; // Squared radius

    // Count neighbors
    int count = 0;
    for (int i = 0; i < N; i++) {
        if (!mask[bs_idx * N + i]) continue;
        
        const float* neighbor = batch_start + i * 3;
        float dx = neighbor[0] - rx;
        float dy = neighbor[1] - ry;
        float dz = neighbor[2] - rz;
        float dist2 = dx*dx + dy*dy + dz*dz;
        if (dist2 <= r2) {
            count++;
        }
    }
    
    // Store density for this point in the batch
    densities[bs_idx * N + pt_idx] = count;
}
*/


/*
__global__ void compute_density_kernel_fast(
    const float *__restrict__ points, 
    int b, int n,
    float radius,
    int* __restrict__ densities,
    int* __restrict__ min_vals,
    int* __restrict__ max_vals,
    const int* __restrict__ bin,
    const bool* __restrict__ mask) 
{

    int bs_idx = blockIdx.y;  // Batch index (0 to B-1)
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x; // Point index in batch
    
    if (bs_idx >= b || pt_idx >= n || !mask[bs_idx * n + pt_idx]) return;
    if (!mask[bs_idx * n + pt_idx]) return; 
    
    // Move pointer to correct batch
    const float* batch_start = points + bs_idx * n * 3;
    const float* current_point = batch_start + pt_idx * 3;
    
    float rx = current_point[0];
    float ry = current_point[1];
    float rz = current_point[2];
    int bin_id = bin[bs_idx * n + pt_idx];
    float r2 = radius * radius; // Squared radius

    // Count neighbors
    int count = 0;
    for (int i = 0; i < n; i++) {
        if (bin_id != bin[bs_idx * n + i]) continue;
        if (!mask[bs_idx * n + i]) continue;
        
        const float* neighbor = batch_start + i * 3;
        float dx = neighbor[0] - rx;
        float dy = neighbor[1] - ry;
        float dz = neighbor[2] - rz;
        float dist2 = dx*dx + dy*dy + dz*dz;
        if (dist2 <= r2) {
            count++;
        }
    }
    
    // Store density for this point in the batch
    densities[bs_idx * n + pt_idx] = count;
    
    atomicMin(&min_vals[bs_idx], count);
    atomicMax(&max_vals[bs_idx], count);
}

void compute_density_kernel_launcher_fast(const float* db_points,
                                          int b, int n,
                                          float radius,
                                          int* densities,
                                          int* min_vals,
                                          int* max_vals,
                                          const int *bin, 
                                          const bool* db_mask) {
                                          
    cudaError_t err;
    
    // Grid configuration
    dim3 blocks(DIVUP(n, THREADS_PER_BLOCK), b);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    // Launch kernel
    compute_density_kernel_fast<<<blocks, threads>>>(db_points,
                                                     b, n,
                                                     radius, 
                                                     densities,
                                                     min_vals,
                                                     max_vals,
                                                     bin,
                                                     db_mask);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
}
*/






__global__ void compute_density_kernel_fast(
    const float *__restrict__ points, 
    int b, int n,
    float radius,
    int* __restrict__ densities,
    int* __restrict__ min_vals,
    int* __restrict__ max_vals,
    const bool* __restrict__ mask) 
{
    int bs_idx = blockIdx.y;  // Batch index (0 to B-1)
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x; // Point index in batch
    
    if (bs_idx >= b || pt_idx >= n || !mask[bs_idx * n + pt_idx]) return;
    if (!mask[bs_idx * n + pt_idx]) return; 
    
    // Move pointer to correct batch
    const float* batch_start = points + bs_idx * n * 3;
    const float* current_point = batch_start + pt_idx * 3;
    
    float rx = current_point[0];
    float ry = current_point[1];
    float rz = current_point[2];
    float r2 = radius * radius; // Squared radius

    // Count neighbors
    int count = 0;
    for (int i = 0; i < n; i++) {
        if (!mask[bs_idx * n + i]) continue;
        
        const float* neighbor = batch_start + i * 3;
        float dx = neighbor[0] - rx;
        float dy = neighbor[1] - ry;
        float dz = neighbor[2] - rz;
        float dist2 = dx*dx + dy*dy + dz*dz;
        if (dist2 <= r2) {
            count++;
        }
    }
    
    // Store density for this point in the batch
    densities[bs_idx * n + pt_idx] = count;
    
    atomicMin(&min_vals[bs_idx], count);
    atomicMax(&max_vals[bs_idx], count);
}

void compute_density_kernel_launcher_fast(const float* db_points,
                                          int b, int n,
                                          float radius,
                                          int* densities,
                                          int* min_vals,
                                          int* max_vals,
                                          const bool* db_mask) {
                                          
    cudaError_t err;
    
    // Grid configuration
    dim3 blocks(DIVUP(n, THREADS_PER_BLOCK), b);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    // Launch kernel
    compute_density_kernel_fast<<<blocks, threads>>>(db_points,
                                                     b, n,
                                                     radius, 
                                                     densities,
                                                     min_vals,
                                                     max_vals,
                                                     db_mask);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
}




















/*
__global__ void compute_density_kernel_fast(const float* __restrict__ query_points,
                                              const float* __restrict__ database_points,
                                              int B, int Nq, int n,
                                              float radius,
                                              int* __restrict__ dq_den, // Database densities
                                              int* __restrict__ min_vals,
                                              int* __restrict__ max_vals,
                                              const bool* query_mask,
                                              const bool* db_mask) {

    int bs_idx = blockIdx.y;  // Batch inex (0 to B-1)
    int q_idx = blockIdx.x * blockDim.x + threadIdx.x;  // Query inex in batch

    if (bs_idx >= B || q_idx >= Nq || !query_mask[bs_idx * Nq + q_idx]) return;

    // Move pointer to correct batch
    const float* q_batch_start = query_points + bs_idx * Nq * 3;
    const float* db_batch_start = database_points + bs_idx * n * 3;
    
    const float* current_query = q_batch_start + q_idx * 3;
    
    float qx = current_query[0];
    float qy = current_query[1];
    float qz = current_query[2];
    float r2 = radius * radius; // Squared radius

    // Shared memory for database points
    __shared__ float shared_db[SHARED_MEM_SIZE][3];
    __shared__ bool shared_mask[SHARED_MEM_SIZE];

    int count = 0;

    // Iterate over database points in chunks
    for (int block_offset = 0; block_offset < n; block_offset += SHARED_MEM_SIZE) {
        int db_idx = block_offset + threadIdx.x;

        if (db_idx < n) {
            shared_db[threadIdx.x][0] = db_batch_start[db_idx * 3 + 0];
            shared_db[threadIdx.x][1] = db_batch_start[db_idx * 3 + 1];
            shared_db[threadIdx.x][2] = db_batch_start[db_idx * 3 + 2];
            shared_mask[threadIdx.x] = db_mask[bs_idx * n + db_idx];
        } else {
            shared_mask[threadIdx.x] = false;
        }
        __syncthreads();

        // Compare the query point to shared database points
        for (int j = 0; j < SHARED_MEM_SIZE && (block_offset + j) < n; ++j) {
            if (!shared_mask[j]) continue;

            float dx = shared_db[j][0] - qx;
            float dy = shared_db[j][1] - qy;
            float dz = shared_db[j][2] - qz;
            float dist2 = dx * dx + dy * dy + dz * dz;

            if (dist2 <= r2) {
                count++;
            }
        }
        __syncthreads();
    }

    // Store query density
    int global_q_idx = bs_idx * Nq + q_idx;
    dq_den[global_q_idx] = count;

    // Use atomic operations to update per-batch min/max
    atomicMin(&min_vals[bs_idx], count);
    atomicMax(&max_vals[bs_idx], count);
}

void compute_density_kernel_launcher_fast(const float* db_points,
                                          int b, int n,
                                          float radius,
                                          int* densities,
                                          int* min_vals,
                                          int* max_vals, 
                                          const bool* db_mask) {
                                          
                                          
    float *dq_points;
    cudaMalloc((void**)&dq_points, b * n * 3 * sizeof(float));
    
    bool *dq_mask;
    cudaMalloc((void**)&dq_mask, b * n * sizeof(bool));
    
    // Copy database points to query points
    cudaMemcpy(dq_points, db_points, b * n * 3 * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(dq_mask, db_mask, b * n * sizeof(bool), cudaMemcpyDeviceToDevice);

    // Grid configuration
    dim3 blocks(DIVUP(n, THREADS_PER_BLOCK), b);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    // Launch kernel
    compute_density_kernel_fast<<<blocks, threads>>>(dq_points, db_points,
                                                     b, n, n,
                                                     radius, 
                                                     densities,
                                                     min_vals,
                                                     max_vals,
                                                     dq_mask,
                                                     db_mask);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
*/

