import torch
import torch.nn as nn
import torch.nn.functional as F
from fast_pytorch_kmeans import KMeans
from typing import Tuple

from .vector_pc_template import VectorFETemplate
from ..vfe import MeanVFE

from ....utils.spconv_utils import spconv
import time

from pcdet.ops.vectornet.vectornet_utils import VectorNetModule

from pytorch3d.ops import knn_points, knn_gather

class NormVecFE(VectorFETemplate):
    def __init__(self, model_cfg, num_point_features, **kwargs):
        super().__init__(model_cfg=model_cfg)
        
        self.mean_vfe = MeanVFE(model_cfg, num_point_features)
        
        self.num_point_features = num_point_features
                
        # self.k = model_cfg.K
        # assert self.k >= 3, "NormVecFE only allows k >= 3"
        
        self.threshold = model_cfg.THRESHOLD
        
        self.vc = VectorNetModule()
        
        self.radius = model_cfg.RADIUS
        self.drop_rate = model_cfg.DROP_RATE
                
        self.if_drop = model_cfg.IF_DROP
        self.if_drop_1 = model_cfg.IF_DROP_1 and self.if_drop
        self.if_drop_2 = model_cfg.IF_DROP_2 and self.if_drop
        
        self.fov_aware = model_cfg.FOV_AWARE and self.if_drop_2
        
    def get_output_feature_dim(self):
        return self.num_point_features

    def forward(self, batch_dict, **kwargs):
        
        batch_size = batch_dict['batch_size']
        
        """
        if 'voxel_features' not in batch_dict:
            batch_dict = self.mean_vfe(batch_dict)
        
        voxels = batch_dict['voxels']
        """
        
        voxel_features = batch_dict['voxel_features']
        voxel_coords = batch_dict['voxel_coords']
        voxel_num_points = batch_dict['voxel_num_points'] 
        
        with torch.no_grad():
            norm_features, mask_1, mask_2 = self.generate_masks(voxel_coords, voxel_features, batch_size)
        
        batch_dict['norm_features'] = norm_features[mask_1][mask_2]
        
        # batch_dict['voxels'] = voxels[mask_1][mask_2]
        
        batch_dict['voxel_features'] = voxel_features[mask_1][mask_2]
        batch_dict['voxel_coords'] = voxel_coords[mask_1][mask_2]
        batch_dict['voxel_num_points'] = voxel_num_points[mask_1][mask_2]
        
        return batch_dict
        
    def generate_masks(self, coords: torch.Tensor, voxel_features: torch.Tensor, batch_size: int):    
        batch_idx = coords[:, 0]
        unique_batches = torch.unique(batch_idx)  
        _device = voxel_features.device
        
        # generate valid voxels and masks
        batched_vf, batched_mask = self.generate_batch_voxels(batch_idx, unique_batches, voxel_features[:, :3], _device)
        
        # retrieve seven_nn informataion of voxels: batched_seven_nn, dist2^2, idx
        batched_seven_nn, _, _ = self.vc.compute_knn(batched_vf, batched_mask) # (B, N, 7, 3)
        
        # normal vector for each group
        batched_normals = self.vc.compute_normals(batched_seven_nn, batched_mask) # (B, N, 3) (0.229 s)
        
        # normal vector density of each norma vector
        batched_density, b_min, b_max = self.vc.compute_density(batched_normals, self.radius, batched_mask)                
        
        # normalizing normal vector density and create mask based on droping value
        batched_norm_density, mask_out = self.vc.compute_norm_mask(batched_density, b_min, b_max, self.drop_rate, self.threshold, batched_mask)
        
        # merging normal vector features
        batched_norm_features = torch.cat([batched_normals, batched_norm_density], dim=2)
        norm_features = batched_norm_features[batched_mask].view(-1, 4)
        
        # droping by normal vector density
        if self.if_drop_1:
            mask_1 = batched_mask & mask_out
            mask_1 = mask_1[batched_mask].view(-1,)
        else:
            mask_1 = torch.ones(batched_mask.sum(), dtype=torch.bool, device=norm_features.device)
            
        
        # dropping by FOV-aware bin-based distance
        mask_2 = self.points_bin_based_drop(voxel_features[mask_1], batch_size, 500) if self.if_drop_2 else torch.ones(mask_1.sum(), dtype=torch.bool, device=norm_features.device)
        
        # mask_2 = torch.rand(mask_1.sum(), device=norm_features.device) > 0.5

        return norm_features, mask_1, mask_2 # , norm_seven_nn, seven_nn

    def points_bin_based_drop(self, pc: torch.Tensor, 
                              batch_size: int,
                              points_per_bin: int = 500,
                              step: int = 10.) -> torch.Tensor:
    
        dist = torch.norm(pc[:, :2], dim=1)
        max_dist = dist.max()
        bin_size = max_dist / step
        
        # bin_edges = torch.arange(0, max_dist, bin_size.item(), device=pc.device)
        bin_edges = torch.arange(0, 30 + bin_size.item(), bin_size.item(), device=pc.device)
        
        bin_edges[-1] = 30  # Ensure the last edge is exactly 30
    
        if self.fov_aware:
            factors = torch.arange(1, 2*len(bin_edges) - 1, 2, device=pc.device).float()
            points_per_bin = points_per_bin * batch_size
            if len(bin_edges) > 2:
                last_factor = ((bin_edges[-1]**2 - bin_edges[-2]**2) / (bin_edges[1]**2 - bin_edges[0]**2)).item()
                factors[-1] = last_factor
        else:
            factors = torch.ones(len(bin_edges), device=pc.device)
            points_per_bin = points_per_bin * 2 * batch_size
    
    
        bin_indices = torch.bucketize(dist, bin_edges, right=False)
        final_mask = torch.zeros(pc.shape[0], dtype=torch.bool, device=pc.device)
        
        
        for i in range(1, len(bin_edges)):
            bin_mask = (bin_indices == i)
            idx_in_bin = torch.nonzero(bin_mask, as_tuple=True)[0]
    
            if idx_in_bin.numel() == 0:
                continue
    
            factor = factors[i - 1]
    
            keep_count = min(int(points_per_bin * factor), idx_in_bin.numel())
            if keep_count > 0:
                perm = torch.randperm(idx_in_bin.numel(), device=pc.device)
                keep_idx = idx_in_bin[perm[:keep_count]]
                final_mask[keep_idx] = True
    
        final_mask |= (bin_indices >= len(bin_edges) - 1)
    
        return final_mask
        
    def generate_batch_voxels(self, batch_idx: torch.Tensor, unique_batches:torch.Tensor, voxel_features: torch.Tensor, device) -> Tuple[torch.Tensor, torch.Tensor]:
        B = len(unique_batches)
        N = 16000
        
        N, D = voxel_features.size()
        batched_voxel_features = torch.full((B, N, D), float('-inf'), dtype=torch.float32, device=device)
        batched_mask = torch.zeros((B, N), dtype=torch.bool, device=device)
        
        for i, batch_id in enumerate(unique_batches):
            v_mask = (batch_idx == batch_id)        
            real_points = voxel_features[v_mask]               

            num_points_i = real_points.shape[0]
            
            batched_voxel_features[i, :num_points_i] = real_points
            batched_mask[i, :num_points_i] = True
            
        return batched_voxel_features, batched_mask
    

class NormVecFEWithoutC(VectorFETemplate):
    def __init__(self, model_cfg, num_point_features, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.num_point_features = num_point_features
                
        self.k = model_cfg.K
        
        assert self.k >= 3, "NormVecFE only allows k >= 3"
        
        self.radius = model_cfg.RADIUS
        self.drop_rate = model_cfg.DROP_RATE
                
        self.if_drop = model_cfg.IF_DROP
        self.if_drop_1 = model_cfg.IF_DROP_1 and self.if_drop
        self.if_drop_2 = model_cfg.IF_DROP_2 and self.if_drop
        
        self.fov_aware = model_cfg.FOV_AWARE
        
    def get_output_feature_dim(self):
        return self.num_point_features

    def forward(self, batch_dict, **kwargs):
        batch_size = batch_dict['batch_size']


        voxel_features = batch_dict['voxel_features'] 
        voxel_coords = batch_dict['voxel_coords']
        voxel_num_points = batch_dict['voxel_num_points']  
        
        with torch.no_grad():
            norm_features, norm_density, mask_1, mask_2 = self.generate_masks(voxel_coords, voxel_features[:, :3], batch_size)
        
        
        batch_dict['norm_features'] = torch.cat([norm_features, norm_density.unsqueeze(1)], dim=1)[mask_1][mask_2]
        batch_dict['norm_density'] = norm_density[mask_1][mask_2]
        batch_dict['voxel_features'] = voxel_features[mask_1][mask_2]
        batch_dict['voxel_coords'] = voxel_coords[mask_1][mask_2]
        batch_dict['voxel_num_points'] = voxel_num_points[mask_1][mask_2]
        
        return batch_dict
        
    def knn(self, xq, xb=None, dist=None):
        if xb is None: 
            xb = xq
            
        bs_scores = self.dist_calculation(xq, xb) if dist is None else dist
        top_scores, top_indices = bs_scores.topk(k=self.k, dim=1, sorted=True, largest=False)
    
        neighbors = xb[top_indices]
    
        return neighbors

    def generate_masks(self, coords: torch.Tensor, points: torch.Tensor, batch_size: int):        
        batch_indices = coords[:, 0]
        
        total_voxels = coords.shape[0]
        device_ = coords.device
        
        batch_norms_list  = []
        mask_1_small_list = [] 
        norm_den_list = [] 
    
        start_idx = 0
                                        
        for b_idx in range(batch_size):
            this_batch = (batch_indices == b_idx)
            num_voxels = this_batch.sum()
            
            batch_points = points[start_idx : start_idx + num_voxels] # drop points later to preserve neighbor features
            dist_points = self.dist_calculation(batch_points)
            neighbor_points = self.knn(batch_points, dist=dist_points)
            batch_norms = self.compute_normals_batch(neighbor_points)

            dist_norms = self.dist_calculation(
                batch_norms
            )            
                
            den = self.compute_point_density(dist_norms)
            den = (den - den.min()) / (den.max() - den.min())  # normalize normal vector density
            
            if self.if_drop_1:                
                mask_1_small = self.norm_density_drop(den, self.drop_rate)                
                mask_1_small_list.append(mask_1_small)
                
            batch_norms_list.append(batch_norms)
            norm_den_list.append(den)

            start_idx += num_voxels
    
        norm_features = torch.cat(batch_norms_list, dim=0)
        mask_1 = torch.cat(mask_1_small_list, dim=0) if self.if_drop_1 else torch.ones(points.shape[0], dtype=torch.bool, device=device_)
        norm_density = torch.cat(norm_den_list, dim=0)
        
        if self.if_drop_2:
            mask_2 = self.points_bin_based_drop(points[mask_1], batch_size) # apply mask_1 before computing to get mask_2
        else:
            mask_2 = torch.ones(mask_1.sum(), dtype=torch.bool, device=device_)
        
        return norm_features, norm_density, mask_1, mask_2
    
    def compute_point_density(self, dist):
        # can also be applied to norm/point clouds density        
        neighbor_counts = torch.sum(dist < self.radius**2, dim=1) - 1  # Subtract 1 to exclude the point itself
    
        return neighbor_counts
        
    def dist_calculation(self, xq, xb=None):
        # return square value
        if xb is None:
            square_val = (xq**2).sum(dim=1)
            return square_val[..., :, None] - 2 * xq @ xq.transpose(-2, -1) + square_val[..., None, :]
            
        return (xq**2).sum(dim=1)[..., :, None] - 2 * xq @ xb.transpose(-2, -1) + (xb**2).sum(dim=1)[..., None, :]
    
    def compute_normals_batch(self, neighbors_points: torch.Tensor) -> torch.Tensor:
        centroids = torch.mean(neighbors_points, dim=1, keepdim=True)  # (N, 1, 3)
        centered = neighbors_points - centroids                        # (N, M, 3)
        
        covariance_matrices = centered.transpose(1, 2) @ centered
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrices)
        normals = eigenvectors[:, :, 0]
    
        mask = (normals[:, 2] < 0)
        normals[mask] *= -1
        
        return normals
        
    def points_bin_based_drop(self, pc: torch.Tensor, 
                              batch_size: int,
                              points_per_bin: int = 500,
                              step: int = 10.) -> torch.Tensor:
    
        dist = torch.norm(pc[:, :2], dim=1)
        max_dist = dist.max()
        bin_size = max_dist / step
        
        bin_edges = torch.arange(0, 30 + bin_size.item(), bin_size.item(), device=pc.device)
        # bin_edges[-1] = 30  # Ensure the last edge is exactly 30
    
        if self.fov_aware:
            factors = torch.arange(1, 2*len(bin_edges) - 1, 2).float()
            points_per_bin = points_per_bin * batch_size
            if len(bin_edges) > 2:
                last_factor = ((bin_edges[-1]**2 - bin_edges[-2]**2) / (bin_edges[1]**2 - bin_edges[0]**2)).item()
                factors[-1] = last_factor
        else:
            factors = torch.ones(len(bin_edges))
            points_per_bin = points_per_bin * 2 * batch_size
    
    
        bin_indices = torch.bucketize(dist, bin_edges, right=False)
        final_mask = torch.zeros(pc.shape[0], dtype=torch.bool, device=pc.device)
        
        for i in range(1, len(bin_edges)):
            bin_mask = (bin_indices == i)
            idx_in_bin = torch.nonzero(bin_mask, as_tuple=True)[0]
    
            if idx_in_bin.numel() == 0:
                continue
    
            factor = factors[i - 1]
    
            keep_count = min(int(points_per_bin * factor), idx_in_bin.numel())
            if keep_count > 0:
                perm = torch.randperm(idx_in_bin.numel(), device=pc.device)
                keep_idx = idx_in_bin[perm[:keep_count]]
                final_mask[keep_idx] = True
    
        final_mask |= (bin_indices >= len(bin_edges) - 1)
    
        return final_mask

        
    def norm_density_drop(self, den: torch.Tensor,
                       drop_rate: float = 0.8) -> torch.Tensor:
                       
        # drop_rate = torch.rand(1).item() * 0.3 + drop_rate
                           
        final_mask = den < 0.7 # assign thresold
    
        false_indices = torch.nonzero(~final_mask, as_tuple=True)[0]
        num_false_to_retain = int(len(false_indices) * drop_rate)
        random_indices = torch.randperm(len(false_indices))[:num_false_to_retain]
        final_mask[false_indices] = True
        final_mask[false_indices[random_indices]] = False  
    
        return final_mask