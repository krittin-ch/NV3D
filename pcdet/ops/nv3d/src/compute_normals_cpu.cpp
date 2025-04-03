#include <iostream>
#include <Eigen/Dense>
#include "compute_normals_cpu.h"

using namespace Eigen;
using namespace std;


void compute_normals_cpu(at::Tensor voxels_tensor, int B, int N, at::Tensor normals_tensor) {
    auto voxel_ptr = voxels_tensor.data_ptr<float>();
    auto normals_ptr = normals_tensor.data_ptr<float>();

    for (int b = 0; b < B; ++b) {
        for (int n = 0; n < N; ++n) {
            // Pointers to the current voxel's neighborhood
            const float* neighbor_ptr = voxel_ptr  + (b * N + n) * 7 * 3;
            float* normal_ptr = normals_ptr + (b * N + n) * 3;

            // Compute centroid
            Vector3f centroid = Vector3f::Zero();
            for (int i = 0; i < 7; ++i) {
                centroid(0) += neighbor_ptr[i * 3 + 0];  // x
                centroid(1) += neighbor_ptr[i * 3 + 1];  // y
                centroid(2) += neighbor_ptr[i * 3 + 2];  // z
            }
            centroid /= 7.0f;

            // Compute covariance matrix
            Matrix3f cov = Matrix3f::Zero();
            for (int i = 0; i < 7; ++i) {
                Vector3f centered(
                    neighbor_ptr[i * 3 + 0] - centroid(0),
                    neighbor_ptr[i * 3 + 1] - centroid(1),
                    neighbor_ptr[i * 3 + 2] - centroid(2)
                );
                cov += centered * centered.transpose();
            }
            cov /= 7.0f;
            
            // Compute eigenvalues & eigenvectors
            SelfAdjointEigenSolver<Matrix3f> solver(cov);
            Vector3f normal = solver.eigenvectors().col(0);  // Largest eigenvalue's eigenvector

            // Ensure consistent normal direction
            if (normal(2) < 0) normal = -normal;

            // Store result
            normal_ptr[0] = normal(0);
            normal_ptr[1] = normal(1);
            normal_ptr[2] = normal(2);
        }
    }
}
