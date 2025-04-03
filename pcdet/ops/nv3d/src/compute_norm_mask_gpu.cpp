#include <torch/serialize/tensor.h>
#include <vector>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "compute_norm_mask_gpu.h"


void compute_norm_mask_wrapper_fast(at::Tensor den_tensor, at::Tensor normalized_density_tensor, 
                                    int B, int N, float drop_rate, float threshold,
                                    at::Tensor min_tensor,
                                    at::Tensor max_tensor,
                                    at::Tensor random_tensor, 
                                    at::Tensor mask_in_tensor,
                                    at::Tensor mask_out_tensor) {

    const int *den = den_tensor.data_ptr<int>();
    const int * min_vals = min_tensor.data_ptr<int>();
    const int * max_vals = max_tensor.data_ptr<int>();
    const bool *mask_in = mask_in_tensor.data_ptr<bool>();
    const float *random_vals = random_tensor.data_ptr<float>();
    
    float *normalized_den = normalized_density_tensor.data_ptr<float>();
    bool *mask_out = mask_out_tensor.data_ptr<bool>();
    
    compute_norm_mask_kernel_launcher_fast(den, normalized_den, B, N, drop_rate, threshold, min_vals, max_vals, random_vals, mask_in, mask_out);
}
