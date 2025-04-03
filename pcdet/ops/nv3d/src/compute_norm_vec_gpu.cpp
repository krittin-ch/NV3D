#include <torch/serialize/tensor.h>
#include <vector>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "compute_seven_nn_gpu.h"


void compute_norm_vec_wrapper_fast(int B, int N, 
                      at::Tensor points_tensor,
                      at::Tensor normals_tensor,
                      at::Tensor normalized_density_tensor, at::Tensor rand_vals_tensor, 
                      float radius, float drop_rate, at::Tensor mask_in_tensor, at::Tensor mask_out_tensor) {


    const float *points = points_tensor.data_ptr<float>();
    float *normals = normals_tensor.data_ptr<float>();
    float *normalized_density = normalized_density_tensor.data_ptr<float>();
    const float *rand_vals = rand_vals_tensor.data_ptr<float>();
    
    const bool *mask_in = mask_in_tensor.data_ptr<bool>();
    bool *mask_out = mask_out_tensor.data_ptr<bool>();

    compute_norm_vec_wrapper_fast(int B, int N, 
                                  const float *points,
                                  float *normals,
                                  float *normalized_density, const float *rand_vals, 
                                  float radius, float drop_rate, const bool *mask_in, bool *mask_out);
}
