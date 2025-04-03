#include <torch/serialize/tensor.h>
#include <torch/extension.h>

#include "compute_seven_nn_gpu.h"
#include "compute_normals_gpu.h"
#include "compute_density_gpu.h"
#include "compute_norm_mask_gpu.h"
#include "compute_norm_vec_gpu.h"

__device__ at::Tensor gather_neighbors(at::Tensor points, at::Tensor idx) {
    // Get tensor dimensions
    at::Device _device = points.device();
    
    int64_t B = points.size(0);  // Batch size
    int64_t N = points.size(1);  // Number of points
    int64_t K = idx.size(2);     // Number of neighbors

    // Create batch indices equivalent to: batch_idx = torch.arange(B, device=points.device).view(-1, 1, 1).expand(-1, N, K)
    //at::Tensor batch_idx = at::arange(B, at::TensorOptions().dtype(at::kLong).device(_device))
    //                      .view({B, 1, 1})
    //                      .expand({B, N, K});
    
    at::Tensor batch_idx = at::arange(B * N * K, at::TensorOptions().dtype(at::kLong).device(_device))
                              .reshape({B, N, K});

    // Perform indexing: equivalent to `points[batch_idx, idx, :]` in Python    
    return points.index({batch_idx, idx, at::indexing::Slice()});
}

__global__ void init_array_kernel(int *arr, int size, int value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        arr[idx] = value;
    }
}

void compute_norm_vec_kernel_launcher_fast(int B, int N, 
                      const float *points,
                      float *normals,
                      float *normalized_density, const float *rand_vals, 
                      float radius, float drop_rate, const bool *mask_in, bool *mask_out,
                      cudaStream_t stream = 0) {  
    
    cudaStream_t stream1, stream2, stream3;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);

    at::Device _device = at::Device(at::kCUDA);

    float *dist2, *neighbors;
    int64_t *idx;
    int *density, *min_vals, *max_vals;

    cudaMalloc((void**)&dist2, B * N * 7 * sizeof(float));
    cudaMalloc((void**)&idx, B * N * 7 * sizeof(int64_t));
    cudaMalloc((void**)&neighbors, B * N * 7 * 3 * sizeof(float));  // Assuming 3D coordinates
    cudaMalloc((void**)&density, B * N * sizeof(int));
    cudaMalloc((void**)&min_vals, B * sizeof(int));
    cudaMalloc((void**)&max_vals, B * sizeof(int));

    cudaMemset(dist2, 0, B * N * 7 * sizeof(float));
    cudaMemset(idx, 0, B * N * 7 * sizeof(int64_t));
    cudaMemset(density, 0, B * N * sizeof(int));

    int block_size = 512;
    int grid_size = (B + block_size - 1) / block_size;
    init_array_kernel<<<grid_size, block_size, 0, stream1>>>(idx, B, 0);
    init_array_kernel<<<grid_size, block_size, 0, stream2>>>(min_vals, B, 50000);
    init_array_kernel<<<grid_size, block_size, 0, stream3>>>(max_vals, B, 0);

    compute_seven_nn_kernel_launcher_fast(B, N, points, dist2, idx, mask_in, stream);

    at::Tensor neighbors_tensor = gather_neighbors(points, idx);  
    cudaMemcpy(neighbors, neighbors_tensor.data_ptr<float>(), B * N * 7 * 3 * sizeof(float), cudaMemcpyDeviceToDevice);

    compute_normals_kernel_launcher_fast(B, N, neighbors, normals, mask_in, stream);
    
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    compute_density_kernel_launcher_fast(B, N, radius, normals, density, min_vals, max_vals, mask_in, stream)
    
    compute_norm_mask_kernel_launcher_fast(B, N, drop_rate, density, 
                                           normalized_density, min_vals, max_vals, 
                                           rand_vals, mask_in, mask_out, stream);

    // Destroy streams after use
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaStreamDestroy(stream3);

    cudaFree(dist2);
    cudaFree(idx);
    cudaFree(neighbors);
    cudaFree(density);
    cudaFree(min_vals);
    cudaFree(max_vals);
}
