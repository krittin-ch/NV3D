#ifndef _COMPUTE_SEVEN_NN_GPU_H
#define _COMPUTE_SEVEN_NN_GPU_H

#include <torch/serialize/tensor.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

void compute_seven_nn_wrapper_fast(at::Tensor points_tensor, int b, int n,
  at::Tensor dist2_tensor, at::Tensor idx_tensor, at::Tensor mask_tensor);

void compute_seven_nn_kernel_launcher_fast(const float *points, int b, int n,
	float *dist2, int64_t *idx, const bool *mask);


/*
void compute_seven_nn_wrapper_fast(at::Tensor points_tensor, int b, int n,
  at::Tensor dist2_tensor, at::Tensor idx_tensor, at::Tensor bin_tensor, at::Tensor mask_tensor);

void compute_seven_nn_kernel_launcher_fast(const float *points, int b, int n,
	float *dist2, int64_t *idx, const int *bin, const bool *mask);
*/

#endif
