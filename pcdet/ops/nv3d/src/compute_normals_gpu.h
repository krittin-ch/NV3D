#ifndef _COMPUTE_NORMALS_GPU_H
#define _COMPUTE_NORMALS_GPU_H

#include <torch/serialize/tensor.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>


void compute_normals_wrapper_fast(at::Tensor voxels_tensor,
                                  int B, int N, int K,
                                  at::Tensor normals_tensor,
                                  at::Tensor mask_tensor);

void compute_normals_kernel_launcher_fast(const float*  voxels,
                                         int B, int N, int K,
                                         float* normals,
                                         const bool* mask);
                                         
/*
void compute_normals_wrapper_fast(at::Tensor voxels_tensor,
                                  int B, int N,
                                  at::Tensor normals_tensor,
                                  at::Tensor bin_tensor,
                                  at::Tensor mask_tensor);

void compute_normals_kernel_launcher_fast(const float*  voxels,
                                         int B, int N,
                                         float* normals,
                                         const int *bin,
                                         const bool* mask);
*/

#endif