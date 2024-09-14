# Copyright (c) Wang, Z
# ------------------------------------------------------------------------
# Modified from StreamPETR (https://github.com/exiawsh/StreamPETR)
# Copyright (c) Shihao Wang
# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
import warnings
import copy
import torch
import torch.nn as nn
from torch.nn import ModuleList
import torch.utils.checkpoint as cp
import numpy as np
import math

from mmcv.cnn import xavier_init, constant_init, kaiming_init
from mmcv.cnn.bricks.transformer import (BaseTransformerLayer,
                                         build_transformer_layer,
                                         build_transformer_layer_sequence,
                                         build_attention,
                                         build_feedforward_network)
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn import build_norm_layer, xavier_init
from mmcv.runner.base_module import BaseModule
from mmcv.cnn.bricks.registry import (ATTENTION,TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import MultiheadAttention
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttnFunction
from mmcv.utils import deprecated_api_warning, ConfigDict

from mmdet.models.utils.builder import TRANSFORMER
from mmdet.models.utils.transformer import inverse_sigmoid

from .attention import FlashMHA


@TRANSFORMER_LAYER.register_module()
class MV2DFusionTransformerDecoderLayer(BaseModule):
    def __init__(self,
                 attn_cfgs=None,
                 ffn_cfgs=dict(
                     type='FFN',
                     embed_dims=256,
                     feedforward_channels=1024,
                     num_fcs=2,
                     ffn_drop=0.,
                     act_cfg=dict(type='ReLU', inplace=True),
                 ),
                 operation_order=None,
                 norm_cfg=dict(type='LN'),
                 init_cfg=None,
                 batch_first=False,
                 with_cp=True,
                 **kwargs):

        deprecated_args = dict(
            feedforward_channels='feedforward_channels',
            ffn_dropout='ffn_drop',
            ffn_num_fcs='num_fcs')
        for ori_name, new_name in deprecated_args.items():
            if ori_name in kwargs:
                warnings.warn(
                    f'The arguments `{ori_name}` in BaseTransformerLayer '
                    f'has been deprecated, now you should set `{new_name}` '
                    f'and other FFN related arguments '
                    f'to a dict named `ffn_cfgs`. ', DeprecationWarning)
                ffn_cfgs[new_name] = kwargs[ori_name]

        super().__init__(init_cfg)

        self.batch_first = batch_first

        attn_ops = ['self_attn', 'cross_attn']
        ops = ['norm', 'ffn'] + attn_ops
        assert set(operation_order) & set(ops) == \
               set(operation_order), f'The operation_order of' \
                                     f' {self.__class__.__name__} should ' \
                                     f'contains all four operation type ' \
                                     f"{ops}, but got {set(operation_order)}"

        num_attn = sum(operation_order.count(x) for x in attn_ops)

        assert num_attn == len(attn_cfgs), f'The length ' \
                                           f'of attn_cfg {num_attn} is ' \
                                           f'not consistent with the number of attention' \
                                           f'in operation_order {operation_order}.'

        self.num_attn = num_attn
        self.operation_order = operation_order
        self.norm_cfg = norm_cfg
        self.pre_norm = operation_order[0] == 'norm'
        self.attentions = ModuleList()

        index = 0
        for operation_name in operation_order:
            if operation_name in attn_ops:
                if 'batch_first' in attn_cfgs[index]:
                    assert self.batch_first == attn_cfgs[index]['batch_first'] or attn_cfgs[index][
                        'type'] == 'PETRMultiheadFlashAttention'
                else:
                    attn_cfgs[index]['batch_first'] = self.batch_first
                attention = build_attention(attn_cfgs[index])
                # Some custom attentions used as `self_attn`
                # or `cross_attn` can have different behavior.
                attention.operation_name = operation_name
                self.attentions.append(attention)
                index += 1

        self.embed_dims = self.attentions[0].embed_dims

        self.ffns = ModuleList()
        num_ffns = operation_order.count('ffn')
        if isinstance(ffn_cfgs, dict):
            ffn_cfgs = ConfigDict(ffn_cfgs)
        if isinstance(ffn_cfgs, dict):
            ffn_cfgs = [copy.deepcopy(ffn_cfgs) for _ in range(num_ffns)]
        assert len(ffn_cfgs) == num_ffns
        for ffn_index in range(num_ffns):
            if 'embed_dims' not in ffn_cfgs[ffn_index]:
                ffn_cfgs[ffn_index]['embed_dims'] = self.embed_dims
            else:
                assert ffn_cfgs[ffn_index]['embed_dims'] == self.embed_dims
            self.ffns.append(
                build_feedforward_network(ffn_cfgs[ffn_index],
                                          dict(type='FFN')))

        self.norms = ModuleList()
        num_norms = operation_order.count('norm')
        for _ in range(num_norms):
            self.norms.append(build_norm_layer(norm_cfg, self.embed_dims)[1])

        self.use_checkpoint = with_cp

    def _forward(self,
                 query,
                 query_pos=None,
                 temp_memory=None,
                 temp_pos=None,
                 feat_flatten_img=None,
                 spatial_flatten_img=None,
                 level_start_index_img=None,
                 pc_range=None,
                 img_metas=None,
                 lidar2img=None,
                 feat_flatten_pts=None,
                 pos_flatten_pts=None,
                 attn_masks=None,
                 query_key_padding_mask=None,
                 key_padding_mask=None,
                 prev_ref_point=None,
                 **kwargs):

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                                                     f'attn_masks {len(attn_masks)} must be equal ' \
                                                     f'to the number of attention in ' \
                                                     f'operation_order {self.num_attn}'

        for layer in self.operation_order:
            if layer == 'self_attn':
                if temp_memory is not None:
                    temp_key = temp_value = torch.cat([query, temp_memory], dim=0)
                    temp_pos = torch.cat([query_pos, temp_pos], dim=0)
                else:
                    temp_key = temp_value = query
                    temp_pos = query_pos
                query = self.attentions[attn_index](
                    query,
                    temp_key,
                    temp_value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=temp_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    **kwargs)

                attn_index += 1
                identity = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == 'cross_attn':
                query = self.attentions[attn_index](
                    query.transpose(0, 1),
                    query_pos.transpose(0, 1),
                    prev_ref_point,
                    feat_flatten_img,
                    spatial_flatten_img,
                    level_start_index_img,
                    pc_range,
                    lidar2img,
                    img_metas,
                    feat_flatten_pts,
                    pos_flatten_pts,
                )
                query = query.transpose(0, 1)

                attn_index += 1
                identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1
            else:
                raise NotImplementedError

        return query

    def forward(self,
                query,
                query_pos=None,
                temp_memory=None,
                temp_pos=None,
                feat_flatten_img=None,
                spatial_flatten_img=None,
                level_start_index_img=None,
                pc_range=None,
                img_metas=None,
                lidar2img=None,
                feat_flatten_pts=None,
                pos_flatten_pts=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                prev_ref_point=None,
                **kwargs
                ):
        """Forward function for `TransformerCoder`.
        Returns:
            Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if self.use_checkpoint and self.training:
            x = cp.checkpoint(
                self._forward,
                query,
                query_pos,
                temp_memory,
                temp_pos,
                feat_flatten_img,
                spatial_flatten_img,
                level_start_index_img,
                pc_range,
                img_metas,
                lidar2img,
                feat_flatten_pts,
                pos_flatten_pts,
                attn_masks,
                query_key_padding_mask,
                key_padding_mask,
                prev_ref_point,
            )
        else:
            x = self._forward(
                query,
                query_pos,
                temp_memory,
                temp_pos,
                feat_flatten_img,
                spatial_flatten_img,
                level_start_index_img,
                pc_range,
                img_metas,
                lidar2img,
                feat_flatten_pts,
                pos_flatten_pts,
                attn_masks,
                query_key_padding_mask,
                key_padding_mask,
                prev_ref_point,
            )
        return x


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class MV2DFusionTransformerDecoder(BaseModule):
    def __init__(self, transformerlayers=None, num_layers=None, init_cfg=None,
                 post_norm_cfg=dict(type='LN'), return_intermediate=False,):
        super().__init__(init_cfg)

        # base transformer decoder
        if isinstance(transformerlayers, dict):
            transformerlayers = [copy.deepcopy(transformerlayers) for _ in range(num_layers)]
        else:
            assert isinstance(transformerlayers, list) and len(transformerlayers) == num_layers

        self.num_layers = num_layers
        self.layers = ModuleList()
        for i in range(num_layers):
            self.layers.append(build_transformer_layer(transformerlayers[i]))
        self.embed_dims = self.layers[0].embed_dims
        self.pre_norm = self.layers[0].pre_norm

        # custom transformer decoder
        self.return_intermediate = return_intermediate
        if post_norm_cfg is not None:
            self.post_norm = build_norm_layer(post_norm_cfg, self.embed_dims)[1]
        else:
            self.post_norm = None

    def forward(self, query, *args, query_pos=None, reference_points=None, dyn_q_coords=None, dyn_q_probs=None,
                dyn_q_mask=None, dyn_q_pos_branch=None, dyn_q_pos_with_prob_branch=None, dyn_q_prob_branch=None,
                **kwargs):
        assert self.return_intermediate
        dyn_q_logits = dyn_q_probs.log()

        intermediate = []
        intermediate_reference_points = [reference_points]
        intermediate_dyn_q_logits = []
        for i, layer in enumerate(self.layers):
            query = layer(query, *args, query_pos=query_pos, prev_ref_point=reference_points, **kwargs)
            if self.post_norm is not None:
                interm_q = self.post_norm(query)
            else:
                interm_q = query

            # get new dyn_q_probs
            dyn_q_logits_res = dyn_q_prob_branch[i](query.transpose(0, 1)[dyn_q_mask])
            dyn_q_logits = dyn_q_logits + dyn_q_logits_res
            dyn_q_probs = dyn_q_logits.softmax(-1)

            # update reference_points
            dyn_q_ref = (dyn_q_probs[:, None] @ dyn_q_coords)[:, 0]
            new_reference_points = reference_points.clone()
            new_reference_points[dyn_q_mask] = dyn_q_ref
            reference_points = new_reference_points

            # update query_pos
            dyn_q_pos = dyn_q_pos_branch(dyn_q_coords.flatten(-2, -1))
            dyn_q_pos = dyn_q_pos_with_prob_branch(dyn_q_pos, dyn_q_probs)
            new_query_pos = query_pos.transpose(0, 1).clone()
            new_query_pos[dyn_q_mask] = dyn_q_pos
            query_pos = new_query_pos.transpose(0, 1)

            if self.return_intermediate:
                intermediate.append(interm_q)
                intermediate_reference_points.append(new_reference_points)
                intermediate_dyn_q_logits.append(dyn_q_logits)

        return torch.stack(intermediate), torch.stack(intermediate_reference_points), torch.stack(
            intermediate_dyn_q_logits)


@TRANSFORMER.register_module()
class MV2DFusionTransformer(BaseModule):

    def __init__(self, encoder=None, decoder=None, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        if encoder is not None:
            self.encoder = build_transformer_layer_sequence(encoder)
        else:
            self.encoder = None
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = self.decoder.embed_dims

    def init_weights(self):
        super().init_weights()
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight is not None and m.weight.dim() > 1:
                xavier_init(m, distribution='uniform')
        self._is_init = True

    def forward(self, tgt, query_pos, attn_masks,
                feat_flatten_img, spatial_flatten_img, level_start_index_img, pc_range, img_metas, lidar2img,
                feat_flatten_pts=None, pos_flatten_pts=None,
                temp_memory=None, temp_pos=None,
                cross_attn_masks=None, reference_points=None,
                dyn_q_coords=None, dyn_q_probs=None, dyn_q_mask=None, dyn_q_pos_branch=None,
                dyn_q_pos_with_prob_branch=None, dyn_q_prob_branch=None,
                ):
        query_pos = query_pos.transpose(0, 1).contiguous()

        if tgt is None:
            tgt = torch.zeros_like(query_pos)
        else:
            tgt = tgt.transpose(0, 1).contiguous()

        if temp_memory is not None:
            temp_memory = temp_memory.transpose(0, 1).contiguous()
            temp_pos = temp_pos.transpose(0, 1).contiguous()

        assert cross_attn_masks is None
        attn_masks = [attn_masks, None]
        # out_dec: [num_layers, num_query, bs, dim]
        out_dec, reference, dyn_q_logits = self.decoder(
            query=tgt,
            query_pos=query_pos,
            temp_memory=temp_memory,
            temp_pos=temp_pos,
            feat_flatten_img=feat_flatten_img,
            spatial_flatten_img=spatial_flatten_img,
            level_start_index_img=level_start_index_img,
            pc_range=pc_range,
            img_metas=img_metas,
            lidar2img=lidar2img,
            feat_flatten_pts=feat_flatten_pts,
            pos_flatten_pts=pos_flatten_pts,
            attn_masks=attn_masks,
            query_key_padding_mask=None,
            key_padding_mask=None,
            reference_points=reference_points,
            dyn_q_coords=dyn_q_coords,
            dyn_q_probs=dyn_q_probs,
            dyn_q_mask=dyn_q_mask,
            dyn_q_pos_branch=dyn_q_pos_branch,
            dyn_q_pos_with_prob_branch=dyn_q_pos_with_prob_branch,
            dyn_q_prob_branch=dyn_q_prob_branch,
        )
        out_dec = out_dec.transpose(1, 2).contiguous()
        return out_dec, reference, dyn_q_logits


@ATTENTION.register_module()
class MixedCrossAttention(BaseModule):
    def __init__(
            self,
            embed_dims=256,
            num_groups=8,
            num_levels=4,
            num_cams=6,
            dropout=0.1,
            num_pts=13,
            im2col_step=64,
            batch_first=True,
            bias=2.,
            bev_norm=1,
            attn_cfg=None,
    ):
        super(MixedCrossAttention, self).__init__()
        self.embed_dims = embed_dims

        # image ca params
        self.num_groups = num_groups
        self.group_dims = (self.embed_dims // self.num_groups)
        self.num_levels = num_levels
        self.num_cams = num_cams
        self.num_pts = num_pts
        self.weights_fc_img = nn.Linear(self.embed_dims, self.num_groups * self.num_levels * num_pts)
        self.output_proj_img = nn.Linear(self.embed_dims, self.embed_dims)
        self.learnable_fc = nn.Linear(self.embed_dims, num_pts * 3)
        self.cam_embed = nn.Sequential(
            nn.Linear(12, self.embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims // 2, self.embed_dims),
            nn.ReLU(inplace=True),
            nn.LayerNorm(self.embed_dims),
        )

        # point cloud ca params
        self.attn = build_attention(attn_cfg)
        self.pts_q_embed = nn.Sequential(
            nn.Linear(13 * 32, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )
        self.pts_k_embed = nn.Sequential(
            nn.Linear(256, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )
        self.weights_fc_pts = nn.Linear(self.embed_dims, num_pts)
        self.pts_q_prob = SELayer_Linear(self.embed_dims, num_pts)

        self.drop = nn.Dropout(dropout)
        self.im2col_step = im2col_step
        self.bias = bias
        self.bev_norm = bev_norm

    def pos2posemb2d(self, pos, num_pos_feats=128, temperature=20):
        scale = 2 * math.pi
        pos = pos * scale
        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
        dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / num_pos_feats)
        pos_x = pos[..., 0, None] / dim_t
        pos_y = pos[..., 1, None] / dim_t
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
        posemb = torch.cat((pos_y, pos_x), dim=-1)
        return posemb

    def init_weights(self):
        nn.init.uniform_(self.learnable_fc.bias.data, -self.bias, self.bias)
        constant_init(self.weights_fc_img, val=0.0, bias=0.0)
        constant_init(self.weights_fc_pts, val=0.0, bias=0.0)
        xavier_init(self.output_proj_img, distribution="uniform", bias=0.0)

    def forward(self, instance_feature, query_pos, reference_points, feat_flatten_img, spatial_flatten_img,
                level_start_index_img, pc_range, lidar2img_mat, img_metas, feat_flatten_pts,
                pos_flatten_pts, ):

        bs, num_anchor = reference_points.shape[:2]

        reference_points = reference_points * (pc_range[3:6] - pc_range[0:3]) + pc_range[0:3]
        key_points = reference_points.unsqueeze(-2) + self.learnable_fc(instance_feature).reshape(bs, num_anchor, -1, 3)

        # image cross-attention
        weights_img = self._get_weights_img(instance_feature, query_pos, lidar2img_mat)
        features_img = self.feature_sampling_img(feat_flatten_img, spatial_flatten_img, level_start_index_img,
                                                 key_points, weights_img, lidar2img_mat, img_metas)
        output = self.output_proj_img(features_img)
        output = self.drop(output) + instance_feature

        # point cloud cross-attention
        weights_pts = self._get_weights_pts(instance_feature, query_pos)
        key_points = (key_points[..., 0:2] - pc_range[0:2]) / (pc_range[3:5] - pc_range[0:2])   # [B, n_q, 13, 2]
        pts_q_pos = self.pts_q_embed(self.pos2posemb2d(key_points, num_pos_feats=16).flatten(-2, -1))
        pts_k_pos = self.pts_k_embed(self.pos2posemb2d(pos_flatten_pts / self.bev_norm, num_pos_feats=128))
        pts_q_pos = self.pts_q_prob(pts_q_pos, weights_pts.flatten(-2, -1))
        output = self.attn(
            output,
            key=feat_flatten_pts,
            value=feat_flatten_pts,
            query_pos=pts_q_pos,
            key_pos=pts_k_pos,)

        return output

    def _get_weights_img(self, instance_feature, anchor_embed, lidar2img_mat, dyn_q_mask=None, dyn_feats=None):
        bs, num_anchor = instance_feature.shape[:2]
        lidar2img = lidar2img_mat[..., :3, :].flatten(-2)
        cam_embed = self.cam_embed(lidar2img)  # B, N, C
        feat_pos_img = (instance_feature + anchor_embed).unsqueeze(2) + cam_embed.unsqueeze(1)
        weights = self.weights_fc_img(feat_pos_img).reshape(bs, num_anchor, -1, self.num_groups).softmax(dim=-2)
        weights = weights.reshape(bs, num_anchor, self.num_cams, -1, self.num_groups).permute(0, 2, 1, 4,
                                                                                              3).contiguous()
        return weights.flatten(end_dim=1)

    def _get_weights_pts(self, instance_feature, anchor_embed):
        bs, num_anchor = instance_feature.shape[:2]
        feat_pos_pts = instance_feature + anchor_embed  # [B, n_q, C]
        weights = self.weights_fc_pts(feat_pos_pts).reshape(bs, num_anchor, self.num_pts, -1).softmax(dim=-2)    # [B, n_q, n_pts, n_groups]
        weights = weights.reshape(bs, num_anchor, self.num_pts, -1).permute(0, 1, 3, 2).contiguous()
        return weights

    def feature_sampling_img(self, feat_flatten, spatial_flatten, level_start_index, key_points, weights, lidar2img_mat,
                         img_metas):
        # key_points: [B, n_q, num_pts, 3]
        # lidar2img_mat: [B, V, 4, 4]
        bs, num_anchor, _ = key_points.shape[:3]

        pts_extand = torch.cat([key_points, torch.ones_like(key_points[..., :1])], dim=-1)
        # points_2d: [B, V, n_q, num_pts, 3]
        points_2d = torch.matmul(lidar2img_mat[:, :, None, None], pts_extand[:, None, ..., None]).squeeze(-1)

        points_2d = points_2d[..., :2] / torch.clamp(points_2d[..., 2:3], min=1e-5)
        points_2d[..., 0:1] = points_2d[..., 0:1] / img_metas[0]['pad_shape'][0][1]
        points_2d[..., 1:2] = points_2d[..., 1:2] / img_metas[0]['pad_shape'][0][0]

        points_2d = points_2d.flatten(end_dim=1)  # [B * V, n_q, num_pts, 2]
        points_2d = points_2d[:, :, None, None, :, :].repeat(1, 1, self.num_groups, self.num_levels, 1, 1)

        bn, num_value, _ = feat_flatten.size()
        feat_flatten = feat_flatten.reshape(bn, num_value, self.num_groups, -1)
        # points_2d: [B * V, n_groups, n_levels, n_q, num_pts, 2]
        # weights: [B * V, n_q, n_groups, n_levels * n_pts]
        output = MultiScaleDeformableAttnFunction.apply(
            feat_flatten, spatial_flatten, level_start_index, points_2d,
            weights, self.im2col_step)

        output = output.reshape(bs, self.num_cams, num_anchor, -1)

        return output.sum(1)


@ATTENTION.register_module()
class PETRMultiheadFlashAttention(BaseModule):
    """A wrapper for ``torch.nn.MultiheadAttention``.
    This module implements MultiheadAttention with identity connection,
    and positional encoding  is also passed as input.
    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        attn_drop (float): A Dropout layer on attn_output_weights.
            Default: 0.0.
        proj_drop (float): A Dropout layer after `nn.MultiheadAttention`.
            Default: 0.0.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        batch_first (bool): When it is True,  Key, Query and Value are shape of
            (batch, n, embed_dim), otherwise (n, batch, embed_dim).
             Default to False.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=dict(type='Dropout', drop_prob=0.),
                 init_cfg=None,
                 batch_first=True,
                 **kwargs):
        super(PETRMultiheadFlashAttention, self).__init__(init_cfg)
        if 'dropout' in kwargs:
            warnings.warn(
                'The arguments `dropout` in MultiheadAttention '
                'has been deprecated, now you can separately '
                'set `attn_drop`(float), proj_drop(float), '
                'and `dropout_layer`(dict) ', DeprecationWarning)
            attn_drop = kwargs['dropout']
            dropout_layer['drop_prob'] = kwargs.pop('dropout')

        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.batch_first = batch_first

        self.attn = FlashMHA(embed_dims, num_heads, attn_drop, dtype=torch.float16, device='cuda',
                                          **kwargs)

        self.proj_drop = nn.Dropout(proj_drop)
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else nn.Identity()

    @deprecated_api_warning({'residual': 'identity'},
                            cls_name='MultiheadAttention')
    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_pos=None,
                attn_mask=None,
                key_padding_mask=None,
                **kwargs):
        """Forward function for `MultiheadAttention`.
        **kwargs allow passing a more general data flow when combining
        with other operations in `transformerlayer`.
        Args:
            query (Tensor): The input query with shape [num_queries, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
                If None, the ``query`` will be used. Defaults to None.
            value (Tensor): The value tensor with same shape as `key`.
                Same in `nn.MultiheadAttention.forward`. Defaults to None.
                If None, the `key` will be used.
            identity (Tensor): This tensor, with the same shape as x,
                will be used for the identity link.
                If None, `x` will be used. Defaults to None.
            query_pos (Tensor): The positional encoding for query, with
                the same shape as `x`. If not None, it will
                be added to `x` before forward function. Defaults to None.
            key_pos (Tensor): The positional encoding for `key`, with the
                same shape as `key`. Defaults to None. If not None, it will
                be added to `key` before forward function. If None, and
                `query_pos` has the same shape as `key`, then `query_pos`
                will be used for `key_pos`. Defaults to None.
            attn_mask (Tensor): ByteTensor mask with shape [num_queries,
                num_keys]. Same in `nn.MultiheadAttention.forward`.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor with shape [bs, num_keys].
                Defaults to None.
        Returns:
            Tensor: forwarded results with shape
            [num_queries, bs, embed_dims]
            if self.batch_first is False, else
            [bs, num_queries embed_dims].
        """

        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None:
            if query_pos is not None:
                # use query_pos if key_pos is not available
                if query_pos.shape == key.shape:
                    key_pos = query_pos
                else:
                    warnings.warn(f'position encoding of key is'
                                  f'missing in {self.__class__.__name__}.')
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        # Because the dataflow('key', 'query', 'value') of
        # ``torch.nn.MultiheadAttention`` is (num_query, batch,
        # embed_dims), We should adjust the shape of dataflow from
        # batch_first (batch, num_query, embed_dims) to num_query_first
        # (num_query ,batch, embed_dims), and recover ``attn_output``
        # from num_query_first to batch_first.
        if self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
        out = self.attn(
            q=query,
            k=key,
            v=value,
            key_padding_mask=None)[0]

        if self.batch_first:
            out = out.transpose(0, 1)

        return identity + self.dropout_layer(self.proj_drop(out))


class SELayer_Linear(BaseModule):
    def __init__(self, channels, in_channels=None, out_channels=None, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        if in_channels is None:
            in_channels = channels
        self.conv_reduce = nn.Linear(in_channels, channels)
        self.act1 = act_layer()
        self.conv_expand = nn.Linear(channels, channels)
        self.gate = gate_layer()
        if out_channels is not None:
            self.conv_last = nn.Sequential(
                nn.Linear(channels, out_channels),
                nn.LayerNorm(out_channels),
                nn.ReLU(inplace=True),
                nn.Linear(out_channels, out_channels)
            )

    def forward(self, x, x_se):
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        out = x * self.gate(x_se)
        if hasattr(self, 'conv_last'):
            out = self.conv_last(out)
        return out