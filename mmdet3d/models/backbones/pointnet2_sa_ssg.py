# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Sequence

import torch
from torch import Tensor, nn

from mmdet3d.models.layers import PointFPModule, build_sa_module
from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType, OptMultiConfig
from .base_pointnet import BasePointNet


@MODELS.register_module()
class PointNet2SASSG(BasePointNet):
    """PointNet2 with Single-scale grouping.

    Args:
        in_channels (int): Input channels of point cloud.
        num_points (tuple[int]): The number of points which each SA
            module samples.
        radius (tuple[float]): Sampling radii of each SA module.
        num_samples (tuple[int]): The number of samples for ball
            query in each SA module.
        sa_channels (tuple[tuple[int]]): Out channels of each mlp in SA module.
        fp_channels (tuple[tuple[int]]): Out channels of each mlp in FP module.
        norm_cfg (dict): Config of normalization layer.
        sa_cfg (dict): Config of set abstraction module, which may contain
            the following keys and values:

            - pool_mod (str): Pool method ('max' or 'avg') for SA modules.
            - use_xyz (bool): Whether to use xyz as a part of features.
            - normalize_xyz (bool): Whether to normalize xyz with radii in
              each SA module.
    """

    def __init__(self,
                 in_channels: int,
                 sa_mask: int = None,
                 clip: float = None,
                 num_points: Sequence[int] = (2048, 1024, 512, 256),
                 radius: Sequence[float] = (0.2, 0.4, 0.8, 1.2),
                 num_samples: Sequence[int] = (64, 32, 16, 16),
                 #fps_sample_range_list: List[int] = None,
                 sa_channels: Sequence[Sequence[int]] = ((64, 64, 128),
                                                         (128, 128, 256),
                                                         (128, 128, 256),
                                                         (128, 128, 256)),
                 fp_channels: Sequence[Sequence[int]] = ((256, 256), (256,
                                                                      256)),
                 norm_cfg: ConfigType = dict(type='BN2d'),
                 sa_cfg: ConfigType = dict(
                     type='PointSAModule',
                     pool_mod='max',
                     use_xyz=True,
                     normalize_xyz=True),
                 init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg=init_cfg)
        self.num_sa = len(sa_channels)
        self.num_fp = len(fp_channels)
        self.sa_mask = sa_mask
        self.clip = clip
        self.in_channels = in_channels

        assert len(num_points) == len(radius) == len(num_samples) == len(
            sa_channels)
        assert len(sa_channels) >= len(fp_channels)

        self.SA_modules = nn.ModuleList()
        sa_in_channel = in_channels - 3  # number of channels without xyz
        skip_channel_list = [sa_in_channel]

        for sa_index in range(self.num_sa):
            cur_sa_mlps = list(sa_channels[sa_index])
            cur_sa_mlps = [sa_in_channel] + cur_sa_mlps
            sa_out_channel = cur_sa_mlps[-1]

            self.SA_modules.append(
                build_sa_module(
                    num_point=num_points[sa_index],
                    radius=radius[sa_index],
                    num_sample=num_samples[sa_index],
                    mlp_channels=cur_sa_mlps,
                    norm_cfg=norm_cfg,
                    #fps_sample_range_list = [-1] if fps_sample_range_list is None else [fps_sample_range_list[sa_index]],
                    cfg=sa_cfg))
            skip_channel_list.append(sa_out_channel)
            sa_in_channel = sa_out_channel

        self.FP_modules = nn.ModuleList()

        fp_source_channel = skip_channel_list.pop()
        fp_target_channel = skip_channel_list.pop()
        for fp_index in range(len(fp_channels)):
            cur_fp_mlps = list(fp_channels[fp_index])
            cur_fp_mlps = [fp_source_channel + fp_target_channel] + cur_fp_mlps
            self.FP_modules.append(PointFPModule(mlp_channels=cur_fp_mlps))
            if fp_index != len(fp_channels) - 1:
                fp_source_channel = cur_fp_mlps[-1]
                fp_target_channel = skip_channel_list.pop()

    def forward(self, points: Tensor, points_nonfps: Tensor = None) -> Dict[str, List[Tensor]]:
        """Forward pass.

        Args:
            points (torch.Tensor): point coordinates with features,
                with shape (B, N, 3 + input_feature_dim).

        Returns:
            dict[str, list[torch.Tensor]]: Outputs after SA and FP modules.

                - fp_xyz (list[torch.Tensor]): The coordinates of
                    each fp features.
                - fp_features (list[torch.Tensor]): The features
                    from each Feature Propagate Layers.
                - fp_indices (list[torch.Tensor]): Indices of the
                    input points.
        """
        xyz, features = self._split_point_feats(points)
        xyz_nonfps, features_nonfps = None, None
        if(points_nonfps!=None):
            xyz_nonfps, features_nonfps = self._split_point_feats(points_nonfps)
            features_nonfps = features_nonfps[:,:self.in_channels-3,:]

        batch, num_points = xyz.shape[:2]
        indices = xyz.new_tensor(range(num_points)).unsqueeze(0).repeat(
            batch, 1).long()

        point_features = features
        sa_xyz = [xyz]
        sa_features = [features[:,:self.in_channels-3,:]]
        sa_indices = [indices]

        for i in range(self.num_sa):
            if(i==0):
                cur_xyz, cur_features, cur_indices = self.SA_modules[i](
                    sa_xyz[i], sa_features[i], xyz_nonfps, features_nonfps)
            else:
                cur_xyz, cur_features, cur_indices = self.SA_modules[i](
                    sa_xyz[i], sa_features[i])
            if(i==self.num_sa-1 and self.sa_mask!=None):
                    #conf = torch.take(sa_features[0][:,1,:], cur_indices.long())
                    conf = torch.take(point_features[:,self.sa_mask,:], cur_indices.long())
                    if(self.clip is not None):
                        conf_high_count = int(self.clip*conf.shape[1])
                        conf_high = conf.sort(-1)[0][:,conf_high_count:conf_high_count+1]
                        conf = torch.clamp( conf, min=None, max=conf_high )
                    conf = conf/conf.mean(-1,True)
                    cur_features = cur_features*conf[:,None,:]
            sa_xyz.append(cur_xyz)
            sa_features.append(cur_features)
            sa_indices.append(
                torch.gather(sa_indices[-1], 1, cur_indices.long()))

        fp_xyz = [sa_xyz[-1]]
        fp_features = [sa_features[-1]]
        fp_indices = [sa_indices[-1]]

        for i in range(self.num_fp):
            fp_features.append(self.FP_modules[i](
                sa_xyz[self.num_sa - i - 1], sa_xyz[self.num_sa - i],
                sa_features[self.num_sa - i - 1], fp_features[-1]))
            fp_xyz.append(sa_xyz[self.num_sa - i - 1])
            fp_indices.append(sa_indices[self.num_sa - i - 1])

        ret = dict(
            fp_xyz=fp_xyz,
            fp_features=fp_features,
            fp_indices=fp_indices,
            sa_xyz=sa_xyz,
            sa_features=sa_features,
            point_features=point_features,
            sa_indices=sa_indices)
        return ret
