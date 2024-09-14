# Copyright (c) OpenMMLab. All rights reserved.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16, force_fp32
from torch.nn.modules.utils import _pair

from mmdet.models.utils import build_linear_layer
from ..builder import QUERY_GENERATORS


@QUERY_GENERATORS.register_module()
class ImageSinglePointQueryGenerator(BaseModule):
    def __init__(self,
                 return_cfg=dict(),

                 with_avg_pool=True,
                 with_cls=False,
                 with_size=False,
                 with_center=True,
                 with_heading=False,
                 with_attr=False,
                 attr_dim=2,
                 roi_feat_size=7,
                 in_channels=256,
                 num_classes=10,

                 reg_class_agnostic=True,
                 reg_predictor_cfg=dict(type='Linear'),
                 cls_predictor_cfg=dict(type='Linear'),
                 extra_encoding=dict(
                     num_layers=2,
                     feat_channels=[512, 256],
                     features=[
                         dict(
                             type='intrinsic',
                             in_channels=16,
                         ), ]
                 ),
                 num_shared_convs=1,
                 num_shared_fcs=1,
                 num_cls_convs=0,
                 num_cls_fcs=0,
                 num_size_convs=0,
                 num_size_fcs=0,
                 num_center_convs=0,
                 num_center_fcs=0,
                 num_heading_convs=0,
                 num_heading_fcs=0,
                 num_attr_convs=0,
                 num_attr_fcs=0,
                 conv_out_channels=256,
                 fc_out_channels=1024,

                 loss_cls=None,
                 conv_cfg=None,
                 norm_cfg=None,
                 init_cfg=None,
                 **kwargs
                 ):
        super(ImageSinglePointQueryGenerator, self).__init__(init_cfg=init_cfg)

        # assert with_center
        self.return_cfg = return_cfg

        self.with_avg_pool = with_avg_pool
        self.with_cls = with_cls
        self.with_size = with_size
        self.with_center = with_center
        self.with_heading = with_heading
        self.with_attr = with_attr
        self.attr_dim = attr_dim
        self.roi_feat_size = _pair(roi_feat_size)
        self.roi_feat_area = self.roi_feat_size[0] * self.roi_feat_size[1]
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.reg_class_agnostic = reg_class_agnostic
        self.reg_predictor_cfg = reg_predictor_cfg
        self.cls_predictor_cfg = cls_predictor_cfg

        self.loss_cls = loss_cls
        if self.loss_cls.use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

        in_channels = self.in_channels
        if self.with_avg_pool:
            self.avg_pool = nn.AvgPool2d(self.roi_feat_size)
        else:
            in_channels *= self.roi_feat_area
        self.debug_imgs = None

        if num_cls_convs > 0 or num_size_convs > 0:
            assert num_shared_fcs == 0
        if not self.with_cls:
            assert num_cls_convs == 0 and num_cls_fcs == 0
        if not self.with_size:
            assert num_size_convs == 0 and num_size_fcs == 0
        if not self.with_heading:
            assert num_heading_convs == 0 and num_heading_fcs == 0
        if not self.with_center:
            assert num_center_convs == 0 and num_center_fcs == 0
        if not self.with_attr:
            assert num_attr_convs == 0 and num_attr_fcs == 0

        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_cls_convs = num_cls_convs
        self.num_cls_fcs = num_cls_fcs
        self.num_size_convs = num_size_convs
        self.num_size_fcs = num_size_fcs
        self.num_center_convs = num_center_convs
        self.num_center_fcs = num_center_fcs
        self.num_heading_convs = num_heading_convs
        self.num_heading_fcs = num_heading_fcs
        self.num_attr_convs = num_attr_convs
        self.num_attr_fcs = num_attr_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.extra_encoding = extra_encoding

        # build prediction layers
        last_layer_dim = self.build_shared_nn()
        self.shared_out_channels = last_layer_dim
        last_layer_dim = self.build_extra_encoding()
        self.shared_out_channels = last_layer_dim
        self.build_branch()
        self.build_predictor()

        self.relu = nn.ReLU(inplace=True)

        if init_cfg is None:
            self.init_cfg = []

            if self.with_cls:
                self.init_cfg += [dict(type='Normal', std=0.01, override=dict(name='fc_cls'))]
            if self.with_center:
                self.init_cfg += [dict(type='Normal', std=0.001, override=dict(name='fc_center'))]
            if self.with_size:
                self.init_cfg += [dict(type='Normal', std=0.001, override=dict(name='fc_size'))]
            if self.with_heading:
                self.init_cfg += [dict(type='Normal', std=0.001, override=dict(name='fc_heading'))]
            if self.with_attr:
                self.init_cfg += [dict(type='Normal', std=0.001, override=dict(name='fc_attr'))]

            self.init_cfg += [
                dict(
                    type='Xavier',
                    distribution='uniform',
                    override=[
                        dict(name='shared_fcs'),
                        dict(name='cls_fcs'),
                        dict(name='size_fcs'),
                        dict(name='heading_fcs'),
                        dict(name='center_fcs'),
                        dict(name='attr_fcs'),
                        # dict(name='extra_enc'),
                    ])
            ]

        self.fp16_enabled = False
        self.batch_split = False

    def build_shared_nn(self):
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        return last_layer_dim

    def build_extra_encoding(self):
        in_channels = self.shared_out_channels
        feat_channels = self.extra_encoding['feat_channels']
        if isinstance(feat_channels, int):
            feat_channels = [feat_channels] * self.extra_encoding['num_layers']
        else:
            assert len(feat_channels) == self.extra_encoding['num_layers']

        for encoding in self.extra_encoding['features']:
            in_channels = in_channels + encoding['in_channels']

        module = []
        assert self.extra_encoding['num_layers'] > 0
        for i in range(self.extra_encoding['num_layers']):
            module.append(nn.Linear(in_channels, feat_channels[i]))
            in_channels = feat_channels[i]
            module.append(nn.GroupNorm(in_channels // 32, in_channels))
            module.append(nn.ReLU(inplace=True))
        module = nn.Sequential(*module)
        self.extra_enc = module

        return feat_channels[-1]

    def build_predictor(self):
        if self.with_cls:
            if self.custom_cls_channels:
                cls_channels = self.loss_cls.get_cls_channels(self.num_classes)
            else:
                cls_channels = self.num_classes + 1
            self.fc_cls = build_linear_layer(
                self.cls_predictor_cfg,
                in_features=self.cls_last_dim,
                out_features=cls_channels)
        if self.with_size:
            out_dim_size = (3 if self.reg_class_agnostic else 3 * self.num_classes)
            self.fc_size = build_linear_layer(
                self.reg_predictor_cfg,
                in_features=self.size_last_dim,
                out_features=out_dim_size)
        if self.with_heading:
            # sin ry, cos ry
            out_dim_heading = 2
            self.fc_heading = build_linear_layer(
                self.reg_predictor_cfg,
                in_features=self.heading_last_dim,
                out_features=out_dim_heading)
        if self.with_center:
            # cx, cy, d
            out_dim_center = 3
            self.fc_center = build_linear_layer(
                self.reg_predictor_cfg,
                in_features=self.center_last_dim,
                out_features=out_dim_center)
        if self.with_attr:
            out_dim_attr = self.attr_dim
            self.fc_attr = build_linear_layer(
                self.reg_predictor_cfg,
                in_features=self.attr_last_dim,
                out_features=out_dim_attr)

    def build_branch(self):
        self.cls_convs, self.cls_fcs, self.cls_last_dim = \
            self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)

        self.size_convs, self.size_fcs, self.size_last_dim = \
            self._add_conv_fc_branch(
                self.num_size_convs, self.num_size_fcs, self.shared_out_channels)

        self.heading_convs, self.heading_fcs, self.heading_last_dim = \
            self._add_conv_fc_branch(
                self.num_heading_convs, self.num_heading_fcs, self.shared_out_channels)

        self.center_convs, self.center_fcs, self.center_last_dim = \
            self._add_conv_fc_branch(
                self.num_center_convs, self.num_center_fcs, self.shared_out_channels)

        self.attr_convs, self.attr_fcs, self.attr_last_dim = \
            self._add_conv_fc_branch(
                self.num_attr_convs, self.num_attr_fcs, self.shared_out_channels)

        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_fcs == 0:
                self.cls_last_dim *= self.roi_feat_area
            if self.num_size_fcs == 0:
                self.size_last_dim *= self.roi_feat_area
            if self.num_heading_fcs == 0:
                self.size_heading_dim *= self.roi_feat_area
            if self.num_center_fcs == 0:
                self.size_center_dim *= self.roi_feat_area
            if self.num_attr_fcs == 0:
                self.size_attr_dim *= self.roi_feat_area

    @property
    def custom_cls_channels(self):
        if self.loss_cls is None:
            return False
        return getattr(self.loss_cls, 'custom_cls_channels', False)

    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,
                            is_shared=False):
        """Add shared or separable branch.

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared
                or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def process_intrins_feat(self, intrinsics, intrins_feat_scale=0.01):
        intrinsics = intrinsics.view(intrinsics.shape[0], 16).float()
        intrinsics = intrinsics * intrins_feat_scale
        return intrinsics

    @torch.no_grad()
    def get_box_params(self, bboxes, intrinsics, extrinsics, min_bbox_size=4):
        intrinsic_list = []
        extrinsic_list = []
        for img_id, (bbox, intrinsic, extrinsic) in enumerate(zip(bboxes, intrinsics, extrinsics)):
            # bbox: [n, (x, y, x, y)], rois_i: [n, c, h, w], intrinsic: [4, 4], extrinsic: [4, 4]
            intrinsic = intrinsic.double().clone()
            extrinsic = extrinsic.double().clone()
            intrinsic = intrinsic.repeat(bbox.shape[0], 1, 1)
            extrinsic = extrinsic.repeat(bbox.shape[0], 1, 1)
            if len(intrinsic) > 0:
                wh_bbox = bbox[:, 2:4] - bbox[:, :2]
                wh_roi = wh_bbox.new_tensor(self.roi_feat_size)
                scale = wh_roi[None] / wh_bbox.clamp_min(min_bbox_size)
                intrinsic[:, :2, 2] = intrinsic[:, :2, 2] - bbox[:, :2] - 0.5 / scale
                intrinsic[:, :2] = intrinsic[:, :2] * scale[..., None]
            intrinsic_list.append(intrinsic)
            extrinsic_list.append(extrinsic)
        intrinsic_list = torch.cat(intrinsic_list, 0)
        extrinsic_list = torch.cat(extrinsic_list, 0)
        return intrinsic_list, extrinsic_list

    def get_output(self, x, convs, fcs):
        for conv in convs:
            x = conv(x)
        if x.dim() > 2:
            if self.with_avg_pool:
                x = self.avg_pool(x)
            x = x.flatten(1)
        for fc in fcs:
            x = self.relu(fc(x))
        return x

    @force_fp32(apply_to=('center_pred', ))
    def center2lidar(self, center_pred, intrinsic, extrinsic):
        center_img = torch.cat([center_pred[:, :2] * center_pred[:, 2:3], center_pred[:, 2:3]], dim=1)
        center_img_hom = torch.cat([center_img, center_img.new_ones([center_img.shape[0], 1])], dim=1)  # [num_rois, 4]
        lidar2img = torch.bmm(intrinsic.double(), extrinsic.double())
        img2lidar = torch.inverse(lidar2img).float()
        center_lidar = torch.bmm(img2lidar, center_img_hom[..., None])[:, :3, 0]
        return center_lidar

    @auto_fp16(apply_to=('x', ))
    def forward(self, x, proposal_list, img_metas, debug_info=None, **kwargs):
        intrinsics, extrinsics = self.get_box_params(proposal_list,
                                                     [img_meta['intrinsics'] for img_meta in img_metas],
                                                     [img_meta['extrinsics'] for img_meta in img_metas])
        extra_feats = dict(intrinsic=self.process_intrins_feat(intrinsics))

        roi_feat, return_feats = self.get_roi_feat(x, extra_feats)
        center_pred, return_feats = self.get_prediction(roi_feat, intrinsics, extrinsics, extra_feats, return_feats)

        return center_pred, return_feats

    def get_roi_feat(self, x, extra_feats=dict()):
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

    def get_prediction(self, x, intrinsics, extrinsics, extra_feats, return_feats):
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

        center_lidar = self.center2lidar(center_pred, intrinsics, extrinsics)
        if self.with_cls and self.with_size:
            bbox_lidar = torch.cat([center_lidar, size_pred,  center_lidar.new_zeros((center_lidar.size(0), 4))], dim=-1)
            bbox_lidar[:, 7] = 1
            return_feats['cls_scores'] = cls_score[:, :self.cls_out_channels]
            return_feats['bbox_preds'] = bbox_lidar

        return center_lidar, return_feats
