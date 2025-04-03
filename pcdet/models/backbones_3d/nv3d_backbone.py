from functools import partial

import torch
import torch.nn as nn

from .spconv_backbone import post_act_block
# from spconv.pytorch import functional as Fsp

# from ...utils.spconv_utils import replace_feature, spconv
from ...utils.spconv_utils import spconv
# from ...utils import common_utils

  
class FeatureEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        
        self.w_q = nn.Sequential(
            nn.Linear(input_dim, 16, bias=False),
            norm_fn(16),   
            nn.ReLU(),
            nn.Linear(16, 32, bias=False),
            norm_fn(32),
            nn.ReLU(),
            nn.Linear(32, 32, bias=False),
        )

        self.w_k = nn.Sequential(
            nn.Linear(input_dim, 16, bias=False),
            norm_fn(16),   
            nn.ReLU(),
            nn.Linear(16, 32, bias=False),
            norm_fn(32),
            nn.ReLU(),
            nn.Linear(32, 32, bias=False),
        )
        
        self.w_v = nn.Sequential(
            nn.Linear(input_dim, 16, bias=False),
            norm_fn(16),   
            nn.ReLU(),
            nn.Linear(16, 32, bias=False),
            norm_fn(32),
            nn.ReLU(),
            nn.Linear(32, 32, bias=False),
        )
        
        self.scale = nn.Parameter(torch.sqrt(torch.tensor(32.0)))
        self.softmax = nn.Softmax(dim=-1)  # Softmax along last dimension

        self.out = nn.Sequential(
            nn.Linear(32, 32, bias=False),
            norm_fn(32),
            nn.ReLU(),
            nn.Linear(32, 16, bias=False),
            norm_fn(16),
            nn.ReLU(),
            nn.Linear(16, output_dim, bias=False),
        )
        
        self.init_weights()
    
    def init_weights(self):
        init_func = nn.init.xavier_normal_
        
        for module_list in [self.w_q, self.w_k, self.w_v, self.out]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    init_func(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
    
    def forward(self, voxel_features, norm_features, coords, batch_size):
        
        Q = self.w_q(norm_features)  # (N, D)
        Q = Q.unsqueeze(2)           # (N, D, 1)
        
        K = self.w_k(voxel_features) # (N, D)
        K = K.unsqueeze(1)           # (N, 1, D)  -> For matrix multiplication
        
        V = self.w_v(voxel_features)
        
        attn_weight = torch.bmm(Q, K) / self.scale # (N, D, D)
        attn_score = self.softmax(attn_weight)  # (N, D, D)
        
        F = torch.bmm(attn_score, V.unsqueeze(-1)).squeeze(-1)  # (N, D)
        
        out = self.out(F)  # (N, output_dim)
        
        return out
 
class NV3DBackBone(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
          
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        num_filters = model_cfg.NUM_FILTERS = model_cfg.NUM_FILTERS
        self.out_features = model_cfg.OUT_FEATURES

        self.feature_fusion = FeatureEncoder(input_channels, input_channels)
        
        # input processing
        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, num_filters[0], 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(num_filters[0]),
            nn.ReLU(),
        )

        block = post_act_block
        
        self.conv1 = spconv.SparseSequential(
            block(num_filters[0], num_filters[0], 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(num_filters[0], num_filters[1], 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2',
                  conv_type='spconv'),
            block(num_filters[1], num_filters[1], 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(num_filters[1], num_filters[1], 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(num_filters[1], num_filters[2], 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3',
                  conv_type='spconv'),
            block(num_filters[2], num_filters[2], 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(num_filters[2], num_filters[2], 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(num_filters[2], num_filters[3], 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4',
                  conv_type='spconv'),
            block(num_filters[3], num_filters[3], 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(num_filters[3], num_filters[3], 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )
        
        
        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(num_filters[3], self.out_features, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(self.out_features),
            nn.ReLU(),
        )
        

        self.backbone_channels = {}
        self.backbone_channels.update({
            'x_conv1': num_filters[0],
            'x_conv2': num_filters[1],
            'x_conv3': num_filters[2],
            'x_conv4': num_filters[3],
        })
        
        self.num_point_features = self.out_features
          
    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        norm_features = batch_dict['norm_features']
        
        batch_size = batch_dict['batch_size']
        
        merged_features = self.feature_fusion(voxel_features, norm_features, voxel_coords, batch_size)        
        
        input_features = spconv.SparseConvTensor(
            features=merged_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        
        input_sp = self.conv_input(input_features)
        
        x_conv1 = self.conv1(input_sp)
        
        x_conv2 = self.conv2(x_conv1)
        
        x_conv3 = self.conv3(x_conv2)
         
        x_conv4 = self.conv4(x_conv3)
        
        out = self.conv_out(x_conv4)
        
        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8,
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            },
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })
              

        return batch_dict