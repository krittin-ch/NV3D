#ifndef COMPUTE_DENSITY_CPU_H
#define COMPUTE_DENSITY_CPU_H

#include <math.h>
#include <stdio.h>
#include <vector>
#include <stdlib.h>
#include <torch/serialize/tensor.h>

void compute_density_cpu(at::Tensor pc_tensor, int N, float radius,at::Tensor den_tensor);

#endif
