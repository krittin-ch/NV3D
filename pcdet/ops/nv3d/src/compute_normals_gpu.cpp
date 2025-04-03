#include <torch/serialize/tensor.h>
#include <vector>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "compute_normals_gpu.h"


void compute_normals_wrapper_fast(at::Tensor voxels_tensor, 
                                  int B, int N, int K,
                                  at::Tensor normals_tensor,
                                  at::Tensor mask_tensor) {
    const float *d_voxels = voxels_tensor.data_ptr<float>();
    float *d_normals = normals_tensor.data_ptr<float>();
    const bool *mask = mask_tensor.data_ptr<bool>();

    compute_normals_kernel_launcher_fast(d_voxels, B, N, K, d_normals, mask);
}

/*
void compute_normals_wrapper_fast(at::Tensor voxels_tensor, 
                                  int B, int N,
                                  at::Tensor normals_tensor,
                                  at::Tensor bin_tensor,
                                  at::Tensor mask_tensor) {
    const float *d_voxels = voxels_tensor.data_ptr<float>();
    float *d_normals = normals_tensor.data_ptr<float>();
    const int *bin = bin_tensor.data_ptr<int>();
    const bool *mask = mask_tensor.data_ptr<bool>();

    compute_normals_kernel_launcher_fast(d_voxels, B, N, d_normals, bin, mask);
}
*/