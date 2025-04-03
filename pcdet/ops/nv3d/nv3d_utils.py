from typing import Tuple

import torch
import torch.nn as nn
from torch.autograd import Function, Variable

from pcdet.ops.nv3d import nv3d_cuda as nv3d


class ComputeKNNGPU(Function):
    @staticmethod
    def forward(ctx, point_clouds: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find the k nearest neighbors of each point in the point cloud.
        :param ctx: PyTorch Autograd context
        :param point_clouds: (B, N, 3) - Batch of point clouds
        :param k: Number of nearest neighbors (Only supports k = 3, 5, 7)
        :param mask: (B, N) - Boolean mask indicating valid points
        :return:
            dist2: (B, N, k) - Squared distances to k nearest neighbors
            idx: (B, N, k) - Indices of k nearest neighbors
        """

        assert point_clouds.is_contiguous(), "Input point_clouds must be contiguous."
        assert mask.is_contiguous(), "Input mask must be contiguous."
        assert point_clouds.shape[0] == mask.shape[0], "Batch sizes must match."
        assert point_clouds.shape[1] == mask.shape[1], "Point count must match."

        _device = point_clouds.device
        B, N, _ = point_clouds.size()

        dist2 = torch.zeros((B, N, 7), dtype=torch.float32, device=_device)
        idx = torch.zeros((B, N, 7), dtype=torch.int64, device=_device)  # Avoid indexing errors

        nv3d.compute_seven_nn_wrapper(point_clouds, B, N, dist2, idx, mask)

        return dist2, idx


    @staticmethod
    def backward(ctx, a=None, b=None):
        return None, None

class ComputeNormalsGPU(Function):
    @staticmethod
    def forward(ctx, neighbors: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Find the seven nearest neighbors of unknown in known
        :param ctx:
        :param neighbors: (B, N, K, 3)
        :return:
            normals: (B, N, 3) normal vectors representing each points/voxels
        """
        assert neighbors.is_contiguous()
        assert mask.is_contiguous()

        assert neighbors.shape[0] == mask.shape[0]
        assert neighbors.shape[1] == mask.shape[1]
             
        _device = neighbors.device
        
        B, N, K, _ = neighbors.size()
        normals = torch.empty((B, N, 3), dtype=torch.float32, device=_device)
        
        nv3d.compute_normals_wrapper(neighbors, B, N, K, normals, mask)                

        return normals

    @staticmethod
    def backward(ctx, a=None, b=None):
        return None, None

class ComputeDensityGPU(Function):
    @staticmethod
    def forward(ctx, points: torch.Tensor, radius: float, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Find the seven nearest neighbors of unknown in known
        :param ctx:
        :param points: (N, 3)
        :return:
            density: (N, 1) normal vectors representing each points/voxels
        """
        assert points.is_contiguous()
        assert mask.is_contiguous()
        
        assert points.shape[0] == mask.shape[0]
        assert points.shape[1] == mask.shape[1]
        
        _device = points.device
        
        B, N, _ = points.size()
        density = torch.empty((B, N, 1), dtype=torch.int32, device=_device)
        min_vals = torch.full((B, 1), 160000, dtype=torch.int32, device=_device)
        max_vals = torch.full((B, 1), 0, dtype=torch.int32, device=_device)
        
        nv3d.compute_density_wrapper(points, B, N, radius, density, min_vals, max_vals, mask)                
        
        return density, min_vals, max_vals

    @staticmethod
    def backward(ctx, a=None, b=None):
        return None, None


class ComputeNormMaskGPU(Function):
    @staticmethod
    def forward(ctx, density: torch.Tensor, min_vals: torch.Tensor, max_vals: torch.Tensor, drop_rate: float, threshold: float, mask_in: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert density.is_contiguous()
        
        assert mask_in.is_contiguous()
        assert density.shape[0] == mask_in.shape[0]
        assert density.shape[1] == mask_in.shape[1]
        assert density.shape[0] == min_vals.shape[0] == max_vals.shape[0]
        
        _device = density.device
        
        B, N, _ = density.size()
        normalized_density = torch.zeros(density.size(), dtype=torch.float32, device=_device) -1
        mask_out = torch.ones(mask_in.size(), dtype=torch.bool, device=_device)
        rand_vals = torch.rand(density.size(), dtype=torch.float32, device=_device)
        
        nv3d.compute_norm_mask_wrapper(density, normalized_density, B, N, drop_rate, threshold, min_vals, max_vals, rand_vals, mask_in, mask_out)
        
        return normalized_density, mask_out
        
        
    @staticmethod
    def backward(ctx, a=None, b=None):
        return None, None

        
class NV3DModule():
    def __init__(self):        

        # Assign GPU versions of functions
        self.compute_knn_gpu = ComputeKNNGPU.apply
        self.compute_normals_gpu = ComputeNormalsGPU.apply
        self.compute_density_gpu = ComputeDensityGPU.apply
        self.compute_norm_mask_gpu = ComputeNormMaskGPU.apply
        
         
    def compute_knn(self, points, mask):
        """
        Find the seven nearest neighbors of points.
        :param points: (B, N, 3)
        :return:
            neighbors: (B, N, 7, 3) points of 7 nearest neighbors
            dist2: Squared distances
            idx: (B, N, 7) index of 7 nearest neighbors
        """
        if points.is_cuda:
            dist2, idx = self.compute_knn_gpu(points, mask)
        else:
            raise NotImplementedError("CPU implementation of compute_seven_nn is not available.")
            # dist2, idx = self.compute_seven_nn_cpu(points)
            
        B, N, _ = points.shape  # (B, N, 3)
        _, _, K = idx.shape      # (B, N, 7)

        batch_idx = torch.arange(B, device=points.device).view(-1, 1, 1)
        batch_idx = batch_idx.expand(-1, N, K)
                
        neighbors = points[batch_idx, idx, :]
        
        return neighbors, dist2, idx

    def compute_normals(self, neighbors, mask):
        """
        Compute normals from nearest neighbors.
        :param neighbors: (B, N, 7, 3)
        :return: normals (B, N, 3)
        """  
        # return self.compute_normals_gpu(neighbors) if neighbors.is_cuda else self.compute_normals_cpu(neighbors)
        
        if neighbors.is_cuda:
            return self.compute_normals_gpu(neighbors, mask)
        else: 
            raise NotImplementedError("CPU implementation of compute_normals is not available.")
            # return self.compute_normals_cpu(neighbors)
        
 
    def compute_density(self, points, radius, mask):
        """
        Compute point density using either KD-tree or CPU method.
        :param points: (B, N, 3)
        :param radius: Search radius
        :param if_kdtree: Whether to use KD-tree (default: True)
        :return: density (B, N)
        """
        
        if points.is_cuda:
            return self.compute_density_gpu(points, radius, mask)
        else: 
            raise NotImplementedError("CPU implementation of compute_density is not available.")
            
    def compute_norm_mask(self, density, min_vals, max_vals, drop_rate, threshold, batched_mask):
        if density.is_cuda:
            return self.compute_norm_mask_gpu(density, min_vals, max_vals, drop_rate, threshold, batched_mask)
        else:
            raise NotImplementedError("CPU implementation of compute_norm_mask is not available.")


