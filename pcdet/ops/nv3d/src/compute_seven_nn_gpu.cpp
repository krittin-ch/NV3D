#include <torch/serialize/tensor.h>
#include <vector>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "compute_seven_nn_gpu.h"


void compute_seven_nn_wrapper_fast(at::Tensor points_tensor, int b, int n,
    at::Tensor dist2_tensor, at::Tensor idx_tensor, at::Tensor mask_tensor) {
    const float *points = points_tensor.data_ptr<float>();
    float *dist2 = dist2_tensor.data_ptr<float>();
    int64_t *idx = idx_tensor.data_ptr<int64_t>();
    const bool *mask = mask_tensor.data_ptr<bool>();

    compute_seven_nn_kernel_launcher_fast(points, b, n, dist2, idx, mask);
}


/*
void compute_seven_nn_wrapper_fast(at::Tensor points_tensor, int b, int n,
    at::Tensor dist2_tensor, at::Tensor idx_tensor, at::Tensor bin_tensor, at::Tensor mask_tensor) {
    const float *points = points_tensor.data_ptr<float>();
    float *dist2 = dist2_tensor.data_ptr<float>();
    int64_t *idx = idx_tensor.data_ptr<int64_t>();
    const int *bin = bin_tensor.data_ptr<int>();
    const bool *mask = mask_tensor.data_ptr<bool>();

    compute_seven_nn_kernel_launcher_fast(points, b, n, dist2, idx, bin, mask);
}
*/