#ifndef _COMPUTE_NORM_MASK_GPU_H
#define _COMPUTE_NORM_MASK_GPU_H

#include <torch/serialize/tensor.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>


void compute_norm_mask_wrapper_fast(at::Tensor den_tensor, at::Tensor normalized_density_tensor, 
                                    int B, int N, float drop_rate, float threshold,
                                    at::Tensor min_tensor,
                                    at::Tensor max_tensor,
                                    at::Tensor random_tensor, 
                                    at::Tensor mask_in_tensor,
                                    at::Tensor mask_out_tensor);
                                    

void compute_norm_mask_kernel_launcher_fast(const int *density, 
                                              float *normalized_density, 
                                              int B, int N, float drop_rate, float threshold,
                                              const int* min_vals, const int* max_vals,
                                              const float *random_vals,
                                              const bool *mask_in, bool *mask_out);


#endif
