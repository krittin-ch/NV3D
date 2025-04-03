#ifndef COMPUTE_NORMALS_CPU_H
#define COMPUTE_NORMALS_CPU_H

#include <iostream>
#include <Eigen/Dense>
#include <torch/serialize/tensor.h>

void compute_normals_cpu(at::Tensor voxels_tensor, int B, int N, at::Tensor normals_tensor);

#endif
