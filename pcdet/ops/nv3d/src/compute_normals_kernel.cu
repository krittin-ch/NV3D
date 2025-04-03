/*
batch version of normal vectors computation is from VectorNet.
Written by Krittin Chaowakarn
All Rights Reserved 2025.
*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cfloat>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include "cuda_utils.h"
#include "compute_normals_gpu.h"

using namespace std;


__device__ void vec_normalize(float v[3]) {
    float norm = sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    if (norm > FLT_EPSILON) {  // Avoid division by near-zero values
        float invNorm = 1.0f / norm;
        v[0] *= invNorm;
        v[1] *= invNorm;
        v[2] *= invNorm;
    }
}

/*
__device__ bool lu_factor3x3(const float A[9],
                             float L[9],
                             float U[9],
                             int   P[3]) {
    // Step 0: Initialize L as Identity, U as 0, P as the identity permutation
    // We’ll fill U in place as we find pivots.
    for (int i = 0; i < 9; i++) {
        L[i] = 0.0f;
        U[i] = 0.0f;
    }
    L[0] = L[4] = L[8] = 1.0f;  // Diagonal of L = 1
    P[0] = 0; P[1] = 1; P[2] = 2;

    // Make a local copy of A so we can pivot rows
    float M[9];
    for (int i = 0; i < 9; i++) {
        M[i] = A[i];
    }

    for (int k = 0; k < 3; k++)
    {
        // 1) Partial pivot: find the largest pivot in column k among rows k..2
        float maxVal = fabsf(M[P[k]*3 + k]);
        int   pivotRow = k;
        for (int r = k+1; r < 3; r++) {
            float val = fabsf(M[P[r]*3 + k]);
            if (val > maxVal) {
                maxVal = val;
                pivotRow = r;
            }
        }

        // 2) If pivotRow != k, swap row indices in permutation
        if (pivotRow != k) {
            int tmp = P[k];
            P[k] = P[pivotRow];
            P[pivotRow] = tmp;
        }

        // 3) Check for near-singular pivot
        float pivotVal = M[P[k]*3 + k];
        if (fabsf(pivotVal) < FLT_EPSILON) {
            // Pivot is zero => singular
            return false;
        }

        // 4) Compute U row k using pivotVal
        U[k*3 + k] = pivotVal;
        for (int j = k+1; j < 3; j++) {
            U[k*3 + j] = M[P[k]*3 + j];
        }

        // 5) Eliminate below pivot => fill L for rows i = k+1..2
        for (int i = k+1; i < 3; i++) {
            float factor = M[P[i]*3 + k] / pivotVal;
            // This is L(i,k)
            L[i*3 + k] = factor;

            // Subtract factor * row_k from row_i in M
            for (int j = k; j < 3; j++) {
                M[P[i]*3 + j] -= factor * M[P[k]*3 + j];
            }
        }
    }

    U[1*3 + 1] = M[P[1]*3 + 1] + U[1*3 + 1]; 

    if (fabsf(U[2*3 + 2]) < FLT_EPSILON) {
        return false;
    }

    return true;
}


__device__ bool lu_solve3x3(const float L[9],
                            const float U[9],
                            const int   P[3],
                            const float b[3],
                            float       x[3]) {
    // 1) Apply permutation to b
    float bp[3];
    bp[0] = b[P[0]];
    bp[1] = b[P[1]];
    bp[2] = b[P[2]];

    // 2) Forward substitution: L * y = bp
    float y[3];
    for (int i = 0; i < 3; i++) {
        float sum = bp[i];
        for (int j = 0; j < i; j++) {
            sum -= L[i*3 + j] * y[j];
        }
        // L(i,i) = 1 => so y[i] = sum
        y[i] = sum; 
    }

    // 3) Back substitution: U * x = y
    for (int i = 2; i >= 0; i--) {
        float sum = y[i];
        for (int j = i+1; j < 3; j++) {
            sum -= U[i*3 + j] * x[j];
        }
        float pivot = U[i*3 + i];
        if (fabsf(pivot) < FLT_EPSILON) {
            return false; // near-singular pivot
        }
        x[i] = sum / pivot;
    }

    return true;
}

__device__ int inverse_iteration(const float A[9], float eigenvector[3]) {
    // 1) Factor A => P, L, U just once
    float L[9], U[9];
    int   P[3];
    
    bool okFactor = lu_factor3x3(A, L, U, P);
    float v[3] = {1.0f, 1.0f, 1.0f};
    vec_normalize(v);
    float w[3];
    
    const int maxIter = 20;
    int count = 0;
    for (int iter = 0; iter < maxIter; iter++) {
        // Solve: A * w = v  =>  w = A^{-1} * v
        bool okSolve = lu_solve3x3(L, U, P, v, w);
        if (!okSolve) {
            break;
        }
        // Normalize
        vec_normalize(w);
        // Update the guess
        v[0] = w[0];
        v[1] = w[1];
        v[2] = w[2];
        
        count++;
    }
    
    eigenvector[0] = v[0];
    eigenvector[1] = v[1];
    eigenvector[2] = v[2];
    
    return count;
}
*/



__device__ bool solve3x3(const float A[9], const float b[3], float x[3]) {
    // Make local copies because we'll modify them.
    float a[9];
    float c[3];
    for (int i = 0; i < 9; i++) {
        a[i] = A[i];
    }
    for (int i = 0; i < 3; i++) {
        c[i] = b[i];
    }
    
    // Eliminate row 1 and row 2 using row 0.
    if (fabsf(a[0]) < FLT_EPSILON) return false;
    float factor = a[3] / a[0];
    a[3] -= factor * a[0];
    a[4] -= factor * a[1];
    a[5] -= factor * a[2];
    c[1] -= factor * c[0];
    
    factor = a[6] / a[0];
    a[6] -= factor * a[0];
    a[7] -= factor * a[1];
    a[8] -= factor * a[2];
    c[2] -= factor * c[0];
    
    // Eliminate row 2 using row 1.
    if (fabsf(a[4]) < FLT_EPSILON) return false;
    factor = a[7] / a[4];
    a[7] -= factor * a[4];
    a[8] -= factor * a[5];
    c[2] -= factor * c[1];
    
    // Check pivot for row 2.
    if (fabsf(a[8]) < FLT_EPSILON) return false;
    
    // Back substitution:
    x[2] = c[2] / a[8];
    x[1] = (c[1] - a[5]*x[2]) / a[4];
    x[0] = (c[0] - a[1]*x[1] - a[2]*x[2]) / a[0];
    
    return true;
}

__device__ int inverse_iteration(const float A[9], float eigenvector[3]) {
    const int maxIter = 20;  // Number of iterations (tune as needed)
    float v[3] = {1.0f, 1.0f, 1.0f};
    vec_normalize(v);  // Normalize initial guess
    float w[3];
    
    int count = 0;
    for (int iter = 0; iter < maxIter; iter++) {
        bool ok = solve3x3(A, v, w);
        if (!ok) break;  // If the system is singular, exit early
        vec_normalize(w);
        // Update the guess:
        v[0] = w[0];
        v[1] = w[1];
        v[2] = w[2];
        
        count++;
    }
    // After iterations, v is our eigenvector (corresponding to the largest eigenvalue of A^{-1}, i.e. smallest eigenvalue of A).
    eigenvector[0] = v[0];
    eigenvector[1] = v[1];
    eigenvector[2] = v[2];
    
    return count;
}


__global__ void compute_normals_kernel_fast(const float* d_voxels, int B, int N, int K, float* d_normals, const bool* mask) {
    // Each voxel has a neighborhood of 7 points, each with 3 coordinates.
    // d_voxels is assumed to be laid out as [B, N, K, 3]
    // d_normals is [B, N, 3]
    
    int bs_idx = blockIdx.y;             // Batch index
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x; // Point index in batch

    if (bs_idx >= B || pt_idx >= N || !mask[bs_idx * N + pt_idx]) return;
    
    int voxel_idx = (bs_idx * N + pt_idx) * K * 3; // Starting index for this voxel's 7 points
    int normal_idx = (bs_idx * N + pt_idx) * 3;      // Starting index for the normal vector
    
    // Compute centroid of the 7 points.
    float centroid[3] = {0.0f, 0.0f, 0.0f};
    for (int i = 0; i < K; i++) {
        centroid[0] += d_voxels[voxel_idx + i * 3 + 0];
        centroid[1] += d_voxels[voxel_idx + i * 3 + 1];
        centroid[2] += d_voxels[voxel_idx + i * 3 + 2];
    }
    
    float cov[9] = {0.0};  // Compute covariance in double precision
    
    float avg_diff[3] = {0.0f, 0.0f, 0.0f};  // Initialize accumulators
    
    for (int i = 0; i < K; i++) {
        float diff[3];
        diff[0] = K * d_voxels[voxel_idx + i * 3 + 0] - centroid[0];
        diff[1] = K * d_voxels[voxel_idx + i * 3 + 1] - centroid[1];
        diff[2] = K * d_voxels[voxel_idx + i * 3 + 2] - centroid[2];
    
        // Accumulate values for averaging
        avg_diff[0] += diff[0];
        avg_diff[1] += diff[1];
        avg_diff[2] += diff[2];
    
        // Outer product diff * diff^T, add to covariance matrix
        cov[0] += diff[0] * diff[0];
        cov[1] += diff[0] * diff[1];
        cov[2] += diff[0] * diff[2];
        cov[3] += diff[1] * diff[0];
        cov[4] += diff[1] * diff[1];
        cov[5] += diff[1] * diff[2];
        cov[6] += diff[2] * diff[0];
        cov[7] += diff[2] * diff[1];
        cov[8] += diff[2] * diff[2];
    }
    
    float normal[3];
    
    int count = inverse_iteration(cov, normal);
    
    if (count == 0) {
        if (fabs(avg_diff[2]) < 1e-6f) {
            normal[0] = 0.0f, normal[1] = 0.0f, normal[2] = 1.0f;
        } else if (fabs(avg_diff[1]) < 1e-6f) {
            normal[0] = 0.0f, normal[1] = 1.0f, normal[2] = 0.0f;
        } else if (fabs(avg_diff[0]) < 1e-6f) {
            normal[0] = 1.0f, normal[1] = 0.0f, normal[2] = 0.0f;
        } else {
            normal[0] = 0.0f, normal[1] = 0.0f, normal[2] = 1.0f;
        }
    } else {
        // Ensure consistent normal direction (for example, flip if z-component is negative).
        if (normal[2] < 0) {
            normal[0] = -normal[0];
            normal[1] = -normal[1];
            normal[2] = -normal[2];
        }
        vec_normalize(normal);
    }    
    
    // Store the computed normal in global memory.
    d_normals[normal_idx + 0] = normal[0];
    d_normals[normal_idx + 1] = normal[1];
    d_normals[normal_idx + 2] = normal[2];
}


void compute_normals_kernel_launcher_fast(const float* d_voxels,
                                          int B, int N, int K,
                                          float* d_normals,
                                          const bool* mask) {
    cudaError_t err;
    
    dim3 blocks(DIVUP(N, THREADS_PER_BLOCK), B);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    compute_normals_kernel_fast<<<blocks, threads>>>(d_voxels, B, N, K, d_normals, mask);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}








/*
__global__ void compute_normals_kernel_fast(const float* d_voxels, int B, int N, float* d_normals, const int *bin, const bool* mask) {
    // Each voxel has a neighborhood of 7 points, each with 3 coordinates.
    // d_voxels is assumed to be laid out as [B, N, 7, 3]
    // d_normals is [B, N, 3]
    
    int bs_idx = blockIdx.y;             // Batch index
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x; // Point index in batch

    if (bs_idx >= B || pt_idx >= N || !mask[bs_idx * N + pt_idx]) return;
    
    int voxel_idx = (bs_idx * N + pt_idx) * 7 * 3; // Starting index for this voxel's 7 points
    int normal_idx = (bs_idx * N + pt_idx) * 3;      // Starting index for the normal vector
    
    // Compute centroid of the 7 points.
    int k = 7;
    float centroid[3] = {0.0f, 0.0f, 0.0f};
    for (int i = 0; i < k; i++) {
        centroid[0] += d_voxels[voxel_idx + i * 3 + 0];
        centroid[1] += d_voxels[voxel_idx + i * 3 + 1];
        centroid[2] += d_voxels[voxel_idx + i * 3 + 2];
    }
    centroid[0] /= k;
    centroid[1] /= k;
    centroid[2] /= k;

    // Compute the covariance matrix. We'll store it in a 3x3 array (row-major).
    float cov[9] = {0.0f};
    for (int i = 0; i < k; i++) {
        float diff[3];
        diff[0] = d_voxels[voxel_idx + i * 3 + 0] - centroid[0];
        diff[1] = d_voxels[voxel_idx + i * 3 + 1] - centroid[1];
        diff[2] = d_voxels[voxel_idx + i * 3 + 2] - centroid[2];
        // Outer product diff * diff^T, add to covariance matrix.
        cov[0] += diff[0] * diff[0];
        cov[1] += diff[0] * diff[1];
        cov[2] += diff[0] * diff[2];
        cov[3] += diff[1] * diff[0];
        cov[4] += diff[1] * diff[1];
        cov[5] += diff[1] * diff[2];
        cov[6] += diff[2] * diff[0];
        cov[7] += diff[2] * diff[1];
        cov[8] += diff[2] * diff[2];
    }
    
    // Use inverse iteration to compute the eigenvector corresponding to the smallest eigenvalue.
    float normal[3];
    inverse_iteration(cov, normal);
    
    // Optionally, normalize the resulting vector (it should be nearly unit length from the algorithm).
    vec_normalize(normal);
    
    // Ensure consistent normal direction (for example, flip if z-component is negative).
    if (normal[2] < 0) {
        normal[0] = -normal[0];
        normal[1] = -normal[1];
        normal[2] = -normal[2];
    }
    
    // Store the computed normal in global memory.
    d_normals[normal_idx + 0] = normal[0];
    d_normals[normal_idx + 1] = normal[1];
    d_normals[normal_idx + 2] = normal[2];
}


void compute_normals_kernel_launcher_fast(const float* d_voxels,
                                          int B, int N,
                                          float* d_normals,
                                          const int *bin,
                                          const bool* mask) {
    cudaError_t err;
    
    dim3 blocks(DIVUP(N, THREADS_PER_BLOCK), B);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    compute_normals_kernel_fast<<<blocks, threads>>>(d_voxels, B, N, d_normals, bin, mask);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}
*/
