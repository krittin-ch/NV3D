#ifndef _COMPUTE_DENSITY_GPU_H
#define _COMPUTE_DENSITY_GPU_H

#include <torch/serialize/tensor.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>


void compute_density_wrapper_fast(at::Tensor pc_tensor, int B, int N, float radius, at::Tensor den_tensor, at::Tensor min_tensor, at::Tensor max_tensor, at::Tensor mask_tensor);

void compute_density_kernel_launcher_fast(const float* point_clouds, int B, int N, float radius, int* densities, int* min_vals, int* max_vals,const bool* mask);

/*
void compute_density_wrapper_fast(at::Tensor pc_tensor, int B, int N, float radius, at::Tensor den_tensor, at::Tensor min_tensor, at::Tensor max_tensor, at::Tensor bin_tensor, at::Tensor mask_tensor);

void compute_density_kernel_launcher_fast(const float* point_clouds, int B, int N, float radius, int* densities, int* min_vals, int* max_vals, const int *bin,const bool* mask);
*/

#endif