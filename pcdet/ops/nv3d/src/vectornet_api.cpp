#include <torch/serialize/tensor.h>
#include <torch/extension.h>

#include "compute_seven_nn_gpu.h"
// #include "compute_five_nn_gpu.h"
// #include "compute_three_nn_gpu.h"
#include "compute_normals_gpu.h"
#include "compute_density_gpu.h"
#include "compute_norm_mask_gpu.h"

// #include "compute_norm_vec_gpu.h"

#include "compute_seven_nn_cpu.h"
#include "compute_normals_cpu.h"
#include "compute_density_cpu.h"
#include "compute_density_grid_cpu.h"

/*
at::Tensor gather_neighbors(at::Tensor points, at::Tensor idx) {
    // Get tensor dimensions
    at::Device _device = points.device();
    
    int64_t B = points.size(0);  // Batch size
    int64_t N = points.size(1);  // Number of points
    int64_t K = idx.size(2);     // Number of neighbors

    // Create batch indices equivalent to: batch_idx = torch.arange(B, device=points.device).view(-1, 1, 1).expand(-1, N, K)
    //at::Tensor batch_idx = at::arange(B, at::TensorOptions().dtype(at::kLong).device(_device))
    //                      .view({B, 1, 1})
    //                      .expand({B, N, K});
    
    at::Tensor batch_idx = at::arange(B * N * K, at::TensorOptions().dtype(at::kLong).device(_device))
                              .reshape({B, N, K});

    // Perform indexing: equivalent to `points[batch_idx, idx, :]` in Python    
    return points.index({batch_idx, idx, at::indexing::Slice()});
}

void compute_norm_vec(int B, int N, 
                      at::Tensor points,
                      at::Tensor normals,
                      at::Tensor normalized_density, at::Tensor rand_vals, 
                      float radius, float drop_rate, at::Tensor mask_in, at::Tensor mask_out) {
                      
    at::Device _device = points.device();
    
    // Step 1: Compute 7 nearest neighbors (GPU)
    at::Tensor dist2 = at::empty({B, N, 7}, at::TensorOptions().dtype(at::kFloat).device(_device)).fill_(0);
    
    at::Tensor idx = at::empty({B, N, 7}, at::TensorOptions().dtype(at::kLong).device(_device)).fill_(0);
    
    compute_seven_nn_wrapper_fast(B, N, points, dist2, idx, mask_in);
    at::Tensor neighbors = gather_neighbors(points, idx);
    
    dist2.reset();
    idx.reset();
    
    
    // Step 2: Compute normals (GPU)
    compute_normals_wrapper_fast(neighbors, B, N, normals, mask_in);
    
    neighbors.reset();
    
    // Step 3: Compute density (GPU)
    at::Tensor density = at::empty({B, N, 1}, at::TensorOptions().dtype(at::kInt).device(_device));
    
    at::Tensor min_vals = at::empty({B, 1}, at::TensorOptions().dtype(at::kInt).device(_device)).fill_(50000);
    
    at::Tensor max_vals = at::empty({B, 1}, at::TensorOptions().dtype(at::kInt).device(_device)).fill_(0);
    
    compute_density_wrapper_fast(normals, B, N, radius, density, min_vals, max_vals, mask_in);
    
    // Step 4: Compute normal mask (GPU)
    compute_norm_mask_wrapper_fast(density, normalized_density, B, N, drop_rate, min_vals, max_vals, rand_vals, mask_in, mask_out);
    
    density.reset();
    min_vals.reset();
    max_vals.reset();
}
*/

/*
at::Tensor gather_neighbors(at::Tensor points, at::Tensor idx) {
    // Get tensor dimensions
    at::Device _device = points.device();
    
    int64_t B = points.size(0);  // Batch size
    int64_t N = points.size(1);  // Number of points
    int64_t K = idx.size(2);     // Number of neighbors

    // Create batch indices equivalent to: batch_idx = torch.arange(B, device=points.device).view(-1, 1, 1).expand(-1, N, K)
    //at::Tensor batch_idx = at::arange(B, at::TensorOptions().dtype(at::kLong).device(_device))
    //                      .view({B, 1, 1})
    //                      .expand({B, N, K});
    
    at::Tensor batch_idx = at::arange(B * N * K, at::TensorOptions().dtype(at::kLong).device(_device))
                              .reshape({B, N, K});

    // Perform indexing: equivalent to `points[batch_idx, idx, :]` in Python    
    return points.index({batch_idx, idx, at::indexing::Slice()});
}

void compute_norm_vec(int B, int N, 
                      at::Tensor points,
                      at::Tensor normals,
                      at::Tensor normalized_density, at::Tensor rand_vals, 
                      float radius, float drop_rate, at::Tensor mask_in, at::Tensor mask_out) {
                      
    at::Device _device = points.device();
    
    at::cuda::CUDAStream stream = at::cuda::getStreamFromPool();
    at::cuda::setCurrentCUDAStream(stream);

    // Step 1: Compute 7 nearest neighbors (GPU)
    at::Tensor dist2 = at::empty({B, N, 7}, at::TensorOptions().dtype(at::kFloat).device(_device)).fill_(0);
    
    at::Tensor idx = at::empty({B, N, 7}, at::TensorOptions().dtype(at::kLong).device(_device)).fill_(0);
    
    compute_seven_nn_wrapper_fast(B, N, points, dist2, idx, mask_in);
    at::Tensor neighbors = gather_neighbors(points, idx);
    
    dist2.reset();
    idx.reset();
    
    
    // Step 2: Compute normals (GPU)
    compute_normals_wrapper_fast(neighbors, B, N, normals, mask_in);
    
    neighbors.reset();
    
    // Step 3: Compute density (GPU)
    at::Tensor density = at::empty({B, N, 1}, at::TensorOptions().dtype(at::kInt).device(_device));
    
    at::Tensor min_vals = at::empty({B, 1}, at::TensorOptions().dtype(at::kInt).device(_device)).fill_(50000);
    
    at::Tensor max_vals = at::empty({B, 1}, at::TensorOptions().dtype(at::kInt).device(_device)).fill_(0);
    
    compute_density_wrapper_fast(normals, B, N, radius, density, min_vals, max_vals, mask_in);
    
    // Step 4: Compute normal mask (GPU)
    compute_norm_mask_wrapper_fast(density, normalized_density, B, N, drop_rate, min_vals, max_vals, rand_vals, mask_in, mask_out);
    
    density.reset();
    min_vals.reset();
    max_vals.reset();
}
*/


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_seven_nn_wrapper", &compute_seven_nn_wrapper_fast, "compute_seven_nn_wrapper_fast");
    // m.def("compute_five_nn_wrapper", &compute_five_nn_wrapper_fast, "compute_five_nn_wrapper_fast");
    // m.def("compute_three_nn_wrapper", &compute_three_nn_wrapper_fast, "compute_three_nn_wrapper_fast");
    m.def("compute_normals_wrapper", &compute_normals_wrapper_fast, "compute_normals_wrapper_fast");
    m.def("compute_density_wrapper", &compute_density_wrapper_fast, "compute_density_wrapper_fast");
    m.def("compute_norm_mask_wrapper", &compute_norm_mask_wrapper_fast, "compute_norm_mask_wrapper_fast");
    // m.def("compute_norm_vec", &compute_norm_vec, "compute_norm_vec");
    
    // m.def("compute_seven_nn_cpu", &compute_seven_nn_cpu, "compute_seven_nn_cpu");
    // m.def("compute_normals_cpu", &compute_normals_cpu, "compute_normals_cpu");
    // m.def("compute_density_cpu", &compute_density_cpu, "compute_density_cpu");
    // m.def("compute_density_grid_cpu", &compute_density_grid_cpu, "compute_density_grid_cpu");
}
