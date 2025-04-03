#include <torch/serialize/tensor.h>
#include <vector>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "compute_density_gpu.h"


void compute_density_wrapper_fast(at::Tensor pc_tensor, int B, int N, float radius, at::Tensor den_tensor, at::Tensor min_tensor, at::Tensor max_tensor, at::Tensor mask_tensor) {
    const float *pc = pc_tensor.data_ptr<float>();
    int *den = den_tensor.data_ptr<int>();
    const bool *mask = mask_tensor.data_ptr<bool>();
    int *min_vals = min_tensor.data_ptr<int>();
    int *max_vals = max_tensor.data_ptr<int>();
    
    compute_density_kernel_launcher_fast(pc, B, N, radius, den, min_vals, max_vals, mask);
}

/*
void compute_density_wrapper_fast(at::Tensor pc_tensor, int B, int N, float radius, at::Tensor den_tensor, at::Tensor min_tensor, at::Tensor max_tensor, at::Tensor bin_tensor, at::Tensor mask_tensor) {
    const float *pc = pc_tensor.data_ptr<float>();
    int *den = den_tensor.data_ptr<int>();
    const bool *mask = mask_tensor.data_ptr<bool>();
    int *min_vals = min_tensor.data_ptr<int>();
    const int *bin = bin_tensor.data_ptr<int>();
    int *max_vals = max_tensor.data_ptr<int>();
    
    compute_density_kernel_launcher_fast(pc, B, N, radius, den, min_vals, max_vals, bin, mask);
}
*/