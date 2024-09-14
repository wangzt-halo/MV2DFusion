# Copyright (c) OpenMMLab. All rights reserved.
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16, force_fp32
from mmdet.core import bbox2roi
from mmdet.models.utils import build_linear_layer

from ..builder import QUERY_GENERATORS
from .image_singple_point_query_generator import ImageSinglePointQueryGenerator


@QUERY_GENERATORS.register_module()
class ImageDistributionQueryGenerator(ImageSinglePointQueryGenerator):
    def __init__(self,
                 prob_bin=50,
                 depth_range=[0.1, 90],
                 x_range=[-10, 10],
                 y_range=[-8, 8],
                 code_size=10,

                 gt_guided_loss=1.0,
                 gt_guided=False,
                 gt_guided_weight=0.2,
                 gt_guided_ratio=0.075,
                 gt_guided_prob=0.5,
                 **kwargs,
                 ):
        super(ImageDistributionQueryGenerator, self).__init__(**kwargs)
        self.prob_bin = prob_bin
        center = np.array(self.roi_feat_size)
        x_bins = np.linspace(center[0] + x_range[0], center[0] + x_range[1], prob_bin)
        y_bins = np.linspace(center[1] + y_range[0], center[1] + y_range[1], prob_bin)
        d_bins = np.linspace(depth_range[0], depth_range[1], prob_bin)
        xyd_bins = torch.tensor(np.stack([x_bins, y_bins, d_bins], axis=-1), dtype=torch.float)
        self.register_buffer('xyd_bins', xyd_bins)

        self.gt_guided = gt_guided
        self.gt_guided_weight = gt_guided_weight
        self.gt_guided_ratio = gt_guided_ratio
        self.gt_guided_prob = gt_guided_prob
        self.gt_guided_loss = gt_guided_loss

        self.code_size = code_size

        self.fc_center = build_linear_layer(
                self.reg_predictor_cfg,
                in_features=self.center_last_dim,
                out_features=prob_bin * 3)

        self.batch_split = True

    @auto_fp16(apply_to=('x', ))
    def forward(self, x, proposal_list, img_metas, debug_info=None, **kwargs):
        n_rois_per_view, n_rois_per_batch = kwargs['n_rois_per_view'], kwargs['n_rois_per_batch']

        intrinsics, extrinsics = self.get_box_params(proposal_list,
                                                     [img_meta['intrinsics'] for img_meta in img_metas],
                                                     [img_meta['extrinsics'] for img_meta in img_metas])
        extra_feats = dict(intrinsic=self.process_intrins_feat(intrinsics))

        qg_args = dict()
        qg_args['rois'] = bbox2roi(proposal_list)
        if self.training:
            qg_args['gt_bboxes'] = kwargs['data']['gt_bboxes']
            qg_args['gt_depths'] = kwargs['data']['depths']

        roi_feat, return_feats = self.get_roi_feat(x, proposal_list, extra_feats)
        center_pred, return_feats = self.get_prediction(roi_feat, intrinsics, extrinsics, extra_feats, return_feats, n_rois_per_batch, qg_args)

        return center_pred, return_feats

    def get_roi_feat(self, x, proposal_list, extra_feats=dict()):
        roi_feat = x
        return_feats = dict()
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)
        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)
            x = x.flatten(1)
            for fc in self.shared_fcs:
                x = self.relu(fc(x))

        # extra encoding
        enc_feat = [x]
        for enc in self.extra_encoding['features']:
            enc_feat.append(extra_feats.get(enc['type']))
        enc_feat = torch.cat(enc_feat, dim=1).clamp(min=-5e3, max=5e3)
        x = self.extra_enc(enc_feat)
        if self.return_cfg.get('enc', False):
            return_feats['enc'] = x

        return x, return_feats

    def get_prediction(self, x, intrinsics, extrinsics, extra_feats, return_feats, n_rois_per_batch, kwargs=dict()):
        x = torch.nan_to_num(x)
        # separate branches
        x_cls = x
        x_center = x
        x_size = x
        x_heading = x
        x_attr = x

        out_dict = {}
        for output in ['cls', 'size', 'heading', 'center', 'attr']:
            out_dict[f'x_{output}'] = self.get_output(eval(f'x_{output}'), getattr(self, f'{output}_convs'),
                                                      getattr(self, f'{output}_fcs'))
            if self.return_cfg.get(output, False):
                return_feats[output] = out_dict[f'x_{output}']

        x_cls = out_dict['x_cls']
        x_center = out_dict['x_center']
        x_size = out_dict['x_size']
        x_heading = out_dict['x_heading']
        x_attr = out_dict['x_attr']

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        size_pred = self.fc_size(x_size) if self.with_size else None
        heading_pred = self.fc_heading(x_heading) if self.with_heading else None
        center_pred = self.fc_center(x_center) if self.with_center else None
        attr_pred = self.fc_attr(x_attr) if self.with_attr else None

        n_rois = center_pred.size(0)
        center_pred = center_pred.view(n_rois, self.prob_bin, 3)
        depth_logits = center_pred[..., 2]
        depth_prob = F.softmax(depth_logits, dim=-1)

        depth_integral = torch.matmul(depth_prob, self.xyd_bins[:, 2:3])
        xy_integral = torch.matmul(depth_prob[:, None], center_pred[..., 0:2])[:, 0]

        center_integral = torch.cat([xy_integral, depth_integral], dim=-1)

        if self.with_cls and self.with_size:
            center_lidar = self.center2lidar(center_integral, intrinsics, extrinsics)
            bbox_lidar = torch.cat([center_lidar, size_pred,  center_lidar.new_zeros((center_lidar.size(0), 4))], dim=-1)
            # sin, cos
            bbox_lidar[:, 7] = 1
            return_feats['cls_scores'] = cls_score[:, :self.cls_out_channels]
            return_feats['bbox_preds'] = bbox_lidar[:, :self.code_size]

        assert ((depth_prob.sum(-1) - 1).abs() < 1e-4).all()
        # use gt centers to guide image query generation
        if self.training and (self.gt_guided or self.gt_guided_loss > 0):
            loss, pred_inds, correct_probs = self.depth_loss(kwargs['gt_bboxes'], kwargs['gt_depths'], kwargs['rois'],
                                                             depth_logits, self.xyd_bins[:, 2])
            if loss is not None:
                return_feats['d_loss'] = loss * self.gt_guided_loss
                depth_prob = depth_prob.clone()
                if self.gt_guided:
                    pred_probs = depth_prob[pred_inds]
                    pred_depths = (pred_probs @ self.xyd_bins[:, 2:3])[:, 0]
                    correct_depths = (correct_probs @ self.xyd_bins[:, 2:3])[:, 0]
                    ratio_mask = (pred_depths - correct_depths).abs() > correct_depths * self.gt_guided_ratio
                    prob_mask = torch.rand(pred_inds.shape) <= self.gt_guided_prob
                    mask = ratio_mask & prob_mask.to(ratio_mask.device)
                    depth_prob[pred_inds[mask]] = (1 - self.gt_guided_weight) * pred_probs[mask] + self.gt_guided_weight * correct_probs[mask]
            else:
                return_feats['d_loss'] = x.sum() * 0

            assert ((depth_prob.sum(-1) - 1).abs() < 1e-4).all()

        depth_sample = self.xyd_bins[None, :, 2].repeat(n_rois, 1)[..., None]   # [n_rois, prob_bin, 1]
        center_sample = torch.cat([center_pred[..., :2], depth_sample], dim=-1)
        total_size = center_sample.size(1)
        center_sample = self.center2lidar(center_sample.flatten(0, 1),
                                          intrinsics.repeat_interleave(total_size, dim=0),
                                          extrinsics.repeat_interleave(total_size, dim=0))
        center_sample = center_sample.view(n_rois, total_size, 3)
        center_sample = torch.cat([center_sample, depth_prob[..., None]], dim=-1)

        center_sample_batch = center_sample.split(n_rois_per_batch, dim=0)

        return_feats['depth_logits'] = depth_logits
        return_feats['depth_bin'] = self.xyd_bins[:, 2]
        return_feats['query_feats'] = x.split(n_rois_per_batch, dim=0)
        return center_sample_batch, return_feats

    @force_fp32()
    def depth_loss(self, gt_bboxes, gt_depths, rois, depth_logits, depth_bin):
        gt_bboxes = sum(gt_bboxes, [])
        gt_depths = sum(gt_depths, [])
        centers = []
        for img_id, (box, d) in enumerate(zip(gt_bboxes, gt_depths)):
            if box.size(0) > 0:
                img_inds = box.new_full((box.size(0), 1), img_id)
                ct = torch.cat([img_inds, box, d[:, None]], dim=-1)
            else:
                ct = box.new_zeros((0, 6))
            centers.append(ct)
        centers = torch.cat(centers, 0)
        ious = self.box_iou(rois[:, 1:5], centers[:, 1:5])
        if ious.numel() > 0:
            in_same_img = rois[:, 0:1] == centers[None, :, 0]
            ious[~in_same_img] = 0
            new_ious = ious.clone()
            iou_thr = 0.6
            new_ious[ious < iou_thr] = 0

            # 1 pred can only be matched with up to 1 gt
            max_ious_for_preds = new_ious.max(dim=1, keepdim=True).values
            new_ious[new_ious < max_ious_for_preds] = 0
            # 1 gt can only be matched with up to 1 pred
            max_ious_for_gts = new_ious.max(dim=0, keepdim=True).values
            new_ious[new_ious < max_ious_for_gts] = 0
            inds = (new_ious >= iou_thr).nonzero()
            pred_inds, gt_inds = inds[..., 0], inds[..., 1]

            # transform depth values to depth bins
            depth_logits_preds = depth_logits[pred_inds]
            depth_gts = centers[:, 5][gt_inds]
            depth_bin_gts = (depth_gts - depth_bin[0]) / (depth_bin[-1] - depth_bin[0]) * (len(depth_bin) - 1)
            depth_bin_gts = depth_bin_gts.clamp(min=0 + 1e-3, max=len(depth_bin) - 1 - 1e-3)
            depth_bin_gts_l = depth_bin_gts.floor().long()
            depth_bin_gts_r = depth_bin_gts_l + 1
            depth_bin_gts_l_w = depth_bin_gts_r - depth_bin_gts
            depth_bin_gts_r_w = 1 - depth_bin_gts_l_w

            # ce loss for depth bins
            ce_loss_l = F.cross_entropy(depth_logits_preds, depth_bin_gts_l, reduction='none') * depth_bin_gts_l_w
            ce_loss_r = F.cross_entropy(depth_logits_preds, depth_bin_gts_r, reduction='none') * depth_bin_gts_r_w
            ce_loss = (ce_loss_l + ce_loss_r).mean()

            loss = ce_loss
            correct_probs = torch.zeros_like(depth_logits_preds)
            correct_probs.scatter_(1, depth_bin_gts_l[:, None], depth_bin_gts_l_w[:, None])
            correct_probs.scatter_(1, depth_bin_gts_r[:, None], depth_bin_gts_r_w[:, None])
            pred_depths = (correct_probs @ depth_bin[:, None])[:, 0]
            assert (correct_probs.sum(-1) == 1).all()
            assert (((pred_depths - depth_gts).abs() < 2e-1) | (depth_gts > depth_bin[-1])).all()
        else:
            loss = None
            pred_inds = None
            correct_probs = None

        return loss, pred_inds, correct_probs

    @staticmethod
    def box_iou(rois_a, rois_b, eps=1e-4):
        rois_a = rois_a[..., None, :]  # [*, n, 1, 4]
        rois_b = rois_b[..., None, :, :]  # [*, 1, m, 4]
        xy_start = torch.maximum(rois_a[..., 0:2], rois_b[..., 0:2])
        xy_end = torch.minimum(rois_a[..., 2:4], rois_b[..., 2:4])
        wh = torch.maximum(xy_end - xy_start, rois_a.new_tensor(0))  # [*, n, m, 2]
        intersect = wh.prod(-1)  # [*, n, m]
        wh_a = rois_a[..., 2:4] - rois_a[..., 0:2]  # [*, m, 1, 2]
        wh_b = rois_b[..., 2:4] - rois_b[..., 0:2]  # [*, 1, n, 2]
        area_a = wh_a.prod(-1)
        area_b = wh_b.prod(-1)
        union = area_a + area_b - intersect
        iou = intersect / (union + eps)
        return iou

