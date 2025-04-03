#ifndef _COMPUTE_SEVEN_NN_CPU_H
#define _COMPUTE_SEVEN_NN_CPU_H

#include <math.h>
#include <stdio.h>
#include <nanoflann.hpp>
#include <vector>
#include <stdlib.h>
#include <torch/serialize/tensor.h>

void compute_seven_nn_cpu(int b, int n, int m, 
                  at::Tensor& unknown_tensor,  // (B, N, 3)
                  at::Tensor& known_tensor,    // (B, M, 3)
                  at::Tensor& dist2_tensor,          // Output: (B, N, 7)
                  at::Tensor& idx_tensor);
                  
#endif
