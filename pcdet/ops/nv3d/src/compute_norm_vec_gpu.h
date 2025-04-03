#ifndef _COMPUTE_NORM_VEC_GPU_H
#define _COMPUTE_NORM_VEC_GPU_H

#include <torch/serialize/tensor.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime_api.h>


void compute_norm_vec_wrapper_fast(int B, int N, 
                                   at::Tensor points,
                                   at::Tensor normals,
                                   at::Tensor normalized_density, at::Tensor rand_vals, 
                                   float radius, float drop_rate, at::Tensor mask_in, at::Tensor mask_out);
                      
void compute_norm_vec_kernel_launcher_fast(int B, int N, 
                                           const float *points,
                                           float *normals,
                                           float *normalized_density, const float *rand_vals, 
                                           float radius, float drop_rate, const bool *mask_in, bool *mask_out) ;




#endif
