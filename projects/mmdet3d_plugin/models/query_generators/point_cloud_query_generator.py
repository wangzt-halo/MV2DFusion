# Copyright (c) OpenMMLab. All rights reserved.
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16, force_fp32
import torch_scatter

from ..builder import QUERY_GENERATORS


@QUERY_GENERATORS.register_module()
class PointCloudQueryGenerator(BaseModule):
    def __init__(self, in_channels=128, hidden_channel=128, pts_use_cat=False,
                 dataset='nuscenes', virtual_voxel_size=None, point_cloud_range=None, head_pc_range=None):
        super(PointCloudQueryGenerator, self).__init__()

        assert dataset in ['nuscenes', 'argov2']

        # a shared convolution
        self.empty_pos = nn.Embedding(100000, 3)
        self.empty_embed = nn.Embedding(1, in_channels)

        self.pre_bev_embed = nn.Sequential(
            nn.Linear(in_channels, hidden_channel),
            nn.LayerNorm(hidden_channel, hidden_channel),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channel, hidden_channel),
            nn.LayerNorm(hidden_channel, hidden_channel),
        )
        self.bev_embed = nn.Identity()
        self.query_embed = nn.Sequential(
            nn.Linear(in_channels, hidden_channel),
            nn.LayerNorm(hidden_channel, hidden_channel),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channel, hidden_channel),
            nn.LayerNorm(hidden_channel, hidden_channel),
        )
        self.query_pred_embed = nn.Sequential(
            nn.Linear(7 * 32 if dataset == 'nuscenes' else 5 * 32, hidden_channel),
            nn.ReLU(),
            nn.Linear(hidden_channel, hidden_channel)
        )

        self.pts_use_cat = pts_use_cat
        if self.pts_use_cat:
            num_cls = 10 if dataset == 'nuscenes' else 26
            self.pts_cat_embed = nn.Embedding(num_cls, hidden_channel)

        self.virtual_voxel_size = virtual_voxel_size
        self.point_cloud_range = point_cloud_range
        self.head_pc_range = head_pc_range

    def init_weights(self):
        super(PointCloudQueryGenerator, self).init_weights()
        nn.init.uniform_(self.empty_pos.weight.data, 0, 1)
        self.empty_pos.weight.requires_grad = False

    @staticmethod
    def pos2embed(pos, num_pos_feats=128, temperature=10000):
        import math
        scale = 2 * math.pi
        pos = pos * scale
        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
        dim_t = 2 * (dim_t // 2) / num_pos_feats + 1
        pos_x = pos[..., None] / dim_t
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
        return pos_x.flatten(-2)

    def forward(self, lidar_feat, lidar_indices, lidar_xyz, query_feat, query_xyz, query_pred, query_cat, batch_size):
        assert not lidar_indices.requires_grad and not lidar_xyz.requires_grad
        assert not any(x.requires_grad for x in query_xyz) and not any(x.requires_grad for x in query_pred) and \
               not any(x.requires_grad for x in query_cat) and not any(x.requires_grad for x in query_feat)

        device = lidar_feat.device
        voxel_size = torch.tensor(self.virtual_voxel_size, device=device)
        pc_range = torch.tensor(self.point_cloud_range, device=device)

        # convert voxels to pillars
        bev_indices, bev_indices_inv = lidar_indices[:, [0, 2, 3]].unique(dim=0,
                                                                          return_inverse=True)  # [n_v, (bid, y, x)]
        bev_lidar_feat = self.pre_bev_embed(lidar_feat) + lidar_feat
        bev_feat = torch_scatter.scatter(bev_lidar_feat, bev_indices_inv, dim=0, reduce='mean')
        bev_xyz = (bev_indices[:, [2, 1]] + 0.5) * voxel_size[None, :2] + pc_range[None, :2]
        lidar_feat = bev_feat
        lidar_indices = bev_indices
        lidar_xyz = bev_xyz

        # generate query content features from 3D detection
        query_feat = [x if len(x) > 0 else x.new_zeros((0, 128)) for x in query_feat]
        query_feat = [self.query_embed(x) + x for x in query_feat]
        query_feat_w_pred = [x + self.query_pred_embed(self.pos2embed(pred, 32, temperature=20)) for x, pred in
                             zip(query_feat, query_pred)]
        if self.pts_use_cat:
            query_feat_w_pred = [x + self.pts_cat_embed(cat) for x, cat in zip(query_feat_w_pred, query_cat)]
        query_feat = query_feat_w_pred
        query_xyz = query_xyz

        feat_size = lidar_feat.size(-1)
        batch_mask = [lidar_indices[:, 0] == b for b in range(batch_size)]
        lidar_size = [m.sum().item() for m in batch_mask]
        max_size = max(lidar_size)

        # pad key/value positions
        head_pc_range = torch.tensor(self.head_pc_range, device=device)
        lidar_xyz = (lidar_xyz[:, :2] - head_pc_range[:2]) / (head_pc_range[3:5] - head_pc_range[:2])
        lidar_pos = lidar_feat.new_zeros([batch_size, max_size, 2]) + 0.5
        pad_size = min(max_size, self.empty_pos.weight.size(0))
        lidar_pos[:, -pad_size:] = self.empty_pos.weight[:pad_size, :2]
        for b in range(batch_size):
            lidar_pos[b, :lidar_size[b]] = lidar_xyz[batch_mask[b]][..., :2]

        # pad key/value features
        lidar_feat_in = lidar_feat.new_zeros([batch_size, max_size, feat_size]) + self.empty_embed.weight
        for b in range(batch_size):
            lidar_feat_in[b, :lidar_size[b]] = lidar_feat[batch_mask[b]]
        lidar_feat = lidar_feat_in

        # pad query positions
        max_size = max(len(x) for x in query_feat)
        query_pos = lidar_feat.new_zeros([batch_size, max_size, 3])
        pad_size = min(max_size, self.empty_pos.weight.size(0))
        query_pos[:, -pad_size:] = self.empty_pos.weight[:pad_size, :3] * (
                head_pc_range[3:6] - head_pc_range[0:3]) + head_pc_range[0:3]
        for b in range(batch_size):
            query_pos[b, :len(query_feat[b])] = query_xyz[b][..., :3]

        # pad query content feats
        query_feat_in = lidar_feat.new_zeros([batch_size, max_size, feat_size]) + self.empty_embed.weight
        for b in range(batch_size):
            if len(query_feat[b]) > 0:
                query_feat_in[b, :len(query_feat[b])] = query_feat[b]
        query_feat = query_feat_in

        return lidar_feat, lidar_pos, query_feat, query_pos