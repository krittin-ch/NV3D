#ifndef COMPUTE_DENSITY_GRID_CPU_H
#define COMPUTE_DENSITY_GRID_CPU_H

#include <iostream>
#include <vector>
#include <cmath>
#include <torch/serialize/tensor.h>

void compute_density_grid_cpu(at::Tensor& pc_tensor, at::Tensor& den_tensor, float radius, float cellSize);

#endif
