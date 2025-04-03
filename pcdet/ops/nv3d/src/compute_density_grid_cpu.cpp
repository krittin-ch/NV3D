#include <iostream>
#include <vector>
#include <cmath>
#include <torch/serialize/tensor.h>

using namespace std;

// Function to calculate density
void compute_density_grid_cpu(at::Tensor& pc_tensor, at::Tensor& den_tensor, float radius, float cellSize) {
    // Ensure the input tensors are on the CPU and have the correct dimensions
    assert(pc_tensor.device().is_cpu() && den_tensor.device().is_cpu());
    assert(pc_tensor.size(0) == den_tensor.size(0));
    assert(pc_tensor.size(1) == 3); // Points should have 3 coordinates (x, y, z)

    // Get raw pointers to the tensor data
    const float* pc_ptr = pc_tensor.data_ptr<float>();
    float* den_ptr = den_tensor.data_ptr<float>();

    // Number of points
    int num_points = pc_tensor.size(0);

    // Precompute squared radius to avoid sqrt in distance calculations
    float r2 = radius * radius;

    // Assign points to grid cells
    // Use a dense grid represented as a vector of vectors
    unordered_map<int, unordered_map<int, unordered_map<int, vector<int>>>> grid;

    for (int i = 0; i < num_points; ++i) {
        int cellX = static_cast<int>(floor(pc_ptr[i * 3] / cellSize));
        int cellY = static_cast<int>(floor(pc_ptr[i * 3 + 1] / cellSize));
        int cellZ = static_cast<int>(floor(pc_ptr[i * 3 + 2] / cellSize));
        grid[cellX][cellY][cellZ].push_back(i);
    }

    // Calculate density for each point
    for (int i = 0; i < num_points; ++i) {
        float px = pc_ptr[i * 3];
        float py = pc_ptr[i * 3 + 1];
        float pz = pc_ptr[i * 3 + 2];

        int cellX = static_cast<int>(floor(px / cellSize));
        int cellY = static_cast<int>(floor(py / cellSize));
        int cellZ = static_cast<int>(floor(pz / cellSize));

        // Check the center cell and all adjacent cells
        for (int dx = -1; dx <= 1; ++dx) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dz = -1; dz <= 1; ++dz) {
                    int neighborX = cellX + dx;
                    int neighborY = cellY + dy;
                    int neighborZ = cellZ + dz;

                    if (grid.find(neighborX) != grid.end() &&
                        grid[neighborX].find(neighborY) != grid[neighborX].end() &&
                        grid[neighborX][neighborY].find(neighborZ) != grid[neighborX][neighborY].end()) {
                        for (int j : grid[neighborX][neighborY][neighborZ]) {
                            // if (i == j) continue; // Skip the same point

                            float qx = pc_ptr[j * 3];
                            float qy = pc_ptr[j * 3 + 1];
                            float qz = pc_ptr[j * 3 + 2];

                            float dx = px - qx;
                            float dy = py - qy;
                            float dz = pz - qz;
                            float dist2 = dx * dx + dy * dy + dz * dz;

                            if (dist2 < r2) {
                                den_ptr[i] += 1.0f;
                            }
                        }
                    }
                }
            }
        }
    }
}


/*
#include <math.h>
#include <stdio.h>
#include <nanoflann.hpp>
#include <vector>
#include <stdlib.h>
#include <torch/serialize/tensor.h>

#include "compute_density_kdtree_cpu.h"

using namespace nanoflann;
using namespace std;


struct PointCloud {
    vector<array<float, 3>> pts;

    inline size_t kdtree_get_point_count() const { return pts.size(); }

    inline float kdtree_get_pt(const size_t idx, int dim) const {
        return pts[idx][dim];
    }

    template <class BBOX>
    bool kdtree_get_bbox(BBOX&) const { return false; }  // Bounding box not needed
};

// KD-tree alias
using KDTree = KDTreeSingleIndexAdaptor<
    L2_Simple_Adaptor<float, PointCloud>,
    PointCloud,
    3,           // Dimensionality
    size_t       // <-- Force size_t (matches long unsigned int)
>;

void compute_density_kdtree_cpu(at::Tensor pc_tensor, int N, float radius, at::Tensor den_tensor) {
    auto pc_ptr = pc_tensor.data_ptr<float>();
    auto den_ptr = den_tensor.data_ptr<float>();

    // Build the Point Cloud structure
    PointCloud cloud;
    cloud.pts.reserve(N);  // Fix: Use `pts` instead of `points`
    for (int i = 0; i < N; i++) {
        cloud.pts.push_back({pc_ptr[i * 3], pc_ptr[i * 3 + 1], pc_ptr[i * 3 + 2]});
    }

    // Build KD-tree
    KDTree tree(3, cloud, KDTreeSingleIndexAdaptorParams(10));
    tree.buildIndex();

    float r2 = radius * radius;  // Squared search radius

    // Compute densities using KD-tree
    vector<ResultItem<size_t, float>> neighbors;
    SearchParameters search_params;  // Fix: Use `SearchParameters`, not `SearchParams`

    for (int n = 0; n < N; n++) {
        float query_pt[3] = {pc_ptr[n * 3], pc_ptr[n * 3 + 1], pc_ptr[n * 3 + 2]};
        
        neighbors.clear();  // Clear previous results
        size_t num_neighbors = tree.radiusSearch(query_pt, r2, neighbors, search_params);  


        den_ptr[n] = static_cast<float>(num_neighbors);
    }

    // Normalize densities
    float min_den = *min_element(den_ptr, den_ptr + N);
    float max_den = *max_element(den_ptr, den_ptr + N);
    float denom = max_den - min_den + 1e-6f;

    for (int n = 0; n < N; n++) {
        den_ptr[n] = (den_ptr[n] - min_den) / denom;
    }
}
*/

