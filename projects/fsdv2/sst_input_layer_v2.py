# move the computation of position embeding and mask in middle_encoder_layer
import math
import numpy as np

import torch
from mmcv.runner import auto_fp16
from torch import nn

from mmdet3d.models import MIDDLE_ENCODERS


@MIDDLE_ENCODERS.register_module()
class PseudoMiddleEncoderForSpconvFSD(nn.Module):

    def __init__(self,):
        super().__init__()

    @auto_fp16(apply_to=('voxel_feat', ))
    def forward(self, voxel_feats, voxel_coors, batch_size=None):
        '''
        Args:
            voxel_feats: shape=[N, C], N is the voxel num in the batch.
            coors: shape=[N, 4], [b, z, y, x]
        Returns:
            feat_3d_dict: contains region features (feat_3d) of each region batching level. Shape of feat_3d is [num_windows, num_max_tokens, C].
            flat2win_inds_list: two dict containing transformation information for non-shifted grouping and shifted grouping, respectively. The two dicts are used in function flat2window and window2flat.
            voxel_info: dict containing extra information of each voxel for usage in the backbone.
        '''

        voxel_info = {}
        voxel_info['voxel_feats'] = voxel_feats
        voxel_info['voxel_coors'] = voxel_coors

        return voxel_info
