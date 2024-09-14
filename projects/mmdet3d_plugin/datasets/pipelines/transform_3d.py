# Copyright (c) Wang, Z
# ------------------------------------------------------------------------
# Modified from FAR3D https://github.com/megvii-research/Far3D
# Copyright (c) 2023 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from PETR (https://github.com/megvii-research/PETR)
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------

import copy
import numpy as np
import mmcv
from mmdet.datasets.builder import PIPELINES
import torch
from PIL import Image
from mmdet3d.core.points import BasePoints, get_points_type


@PIPELINES.register_module()
class PadMultiViewImage():
    """Pad the multi-view image.
    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",
    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value, 0 by default.
    """
    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        assert size is not None or size_divisor is not None
        assert size_divisor is None or size is None
    
    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        if self.size is not None:
            padded_img = [mmcv.impad(img,
                                shape = self.size, pad_val=self.pad_val) for img in results['img']]
        elif self.size_divisor is not None:
            padded_img = [mmcv.impad_to_multiple(img,
                                self.size_divisor, pad_val=self.pad_val) for img in results['img']]
        results['img_shape'] = [img.shape for img in results['img']]
        results['img'] = padded_img
        results['pad_shape'] = [img.shape for img in padded_img]
        results['pad_fix_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor
    
    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        self._pad_img(results)
        return results


    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_val={self.pad_val})'
        return repr_str


@PIPELINES.register_module()
class NormalizeMultiviewImage(object):
    """Normalize the image.
    Added key is "img_norm_cfg".
    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function to normalize images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        results['img'] = [mmcv.imnormalize(
            img, self.mean, self.std, self.to_rgb) for img in results['img']]
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})'
        return repr_str


@PIPELINES.register_module()
class AV2LoadPointsFromFile(object):

    def __init__(self,
                 coord_type,
                 load_dim=6,
                 use_dim=[0, 1, 2],
                 shift_height=False,
                 use_color=False,
                 file_client_args=dict(backend='disk')):
        self.shift_height = shift_height
        self.use_color = use_color
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert max(use_dim) < load_dim, \
            f'Expect all used dimensions < {load_dim}, got {use_dim}'
        assert coord_type in ['CAMERA', 'LIDAR', 'DEPTH']

        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def _load_points(self, pts_filename):
        from av2.utils.io import read_feather
        lidar = read_feather(pts_filename)
        lidar = lidar.loc[:, ['x', 'y', 'z', 'intensity']].to_numpy().astype(np.float32)
        return lidar

    def __call__(self, results):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data.
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        pts_filename = results['pts_filename']
        points = self._load_points(pts_filename)
        points = points.reshape(-1, self.load_dim)
        points = points[:, self.use_dim]
        attribute_dims = None

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate(
                [points[:, :3],
                 np.expand_dims(height, 1), points[:, 3:]], 1)
            attribute_dims = dict(height=3)

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(color=[
                    points.shape[1] - 3,
                    points.shape[1] - 2,
                    points.shape[1] - 1,
                ]))

        points_class = get_points_type(self.coord_type)
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims)
        results['points'] = points

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__ + '('
        repr_str += f'shift_height={self.shift_height}, '
        repr_str += f'use_color={self.use_color}, '
        repr_str += f'file_client_args={self.file_client_args}, '
        repr_str += f'load_dim={self.load_dim}, '
        repr_str += f'use_dim={self.use_dim})'
        return repr_str


@PIPELINES.register_module()
class AV2LoadMultiViewImageFromFiles(object):

    def __init__(self, to_float32=False, color_type='unchanged'):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        filename = results['img_filename']
        img = [mmcv.imread(name, self.color_type).astype(np.float32) for name in filename]

        results['filename'] = [str(name) for name in filename]
        results['img'] = img
        results['img_shape'] = img[0].shape
        results['ori_lidar2img'] = copy.deepcopy(results['lidar2img'])
        results['scale_factor'] = 1.0
        num_channels = 3
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}')"
        return repr_str


@PIPELINES.register_module()
class AV2ResizeCropFlipRotImageV2():
    def __init__(self, data_aug_conf=None, multi_stamps=False, training=False):
        self.data_aug_conf = data_aug_conf
        self.min_size = 2.0
        self.multi_stamps = multi_stamps
        self.training = training

    def __call__(self, results):
        imgs = results['img']
        N = len(imgs) // 2 if self.multi_stamps else len(imgs)
        new_imgs = []
        new_gt_bboxes = []
        new_centers2d = []
        new_gt_labels = []
        new_depths = []
        ida_mats = []

        with_depthmap = ('depthmap' in results.keys())
        if with_depthmap:
            new_depthmaps = []

        assert self.data_aug_conf['rot_lim'] == (0.0, 0.0), "Rotation is not currently supported"

        for i in range(N):
            H, W = imgs[i].shape[:2]
            if H > W:
                resize, resize_dims, crop = self._sample_augmentation_f(imgs[i])
                img = Image.fromarray(np.uint8(imgs[i]))
                if with_depthmap:
                    depthmap_im = Image.fromarray(results['depthmap'][i])
                else:
                    depthmap_im = None
                img, ida_mat_f, depthmap = self._img_transform(img, resize=resize, resize_dims=resize_dims, crop=crop,
                                                               depthmap=depthmap_im)
                img_f = np.array(img).astype(np.float32)
                if 'gt_bboxes' in results.keys() and len(results['gt_bboxes']) > 0:
                    gt_bboxes = results['gt_bboxes'][i]
                    centers2d = results['centers2d'][i]
                    gt_labels = results['gt_labels'][i]
                    depths = results['depths'][i]
                    if len(gt_bboxes) != 0:
                        gt_bboxes, centers2d, gt_labels, depths = self._bboxes_transform(
                            img_f, gt_bboxes, centers2d, gt_labels, depths, resize=resize, crop=crop,
                        )
                resize, resize_dims, crop, flip, rotate = self._sample_augmentation(img_f)
                img = Image.fromarray(np.uint8(img_f))
                img, ida_mat, depthmap = self._img_transform(img, resize=resize, resize_dims=resize_dims, crop=crop,
                                                             flip=flip, rotate=rotate, depthmap=depthmap)
                img = np.array(img).astype(np.float32)
                if 'gt_bboxes' in results.keys() and len(results['gt_bboxes']) > 0:
                    if len(gt_bboxes) != 0:
                        gt_bboxes, centers2d, gt_labels, depths = self._bboxes_transform(img, gt_bboxes, centers2d,
                                                                                         gt_labels, depths,
                                                                                         resize=resize, crop=crop,
                                                                                         flip=flip, )
                    if len(gt_bboxes) != 0:
                        gt_bboxes, centers2d, gt_labels, depths = self._filter_invisible(img, gt_bboxes, centers2d,
                                                                                         gt_labels, depths)

                    new_gt_bboxes.append(gt_bboxes)
                    new_centers2d.append(centers2d)
                    new_gt_labels.append(gt_labels)
                    new_depths.append(depths)

                new_imgs.append(img)
                results['intrinsics'][i][:3, :3] = ida_mat @ ida_mat_f @ results['intrinsics'][i][:3, :3]
                ida_mats.append(np.array(ida_mat @ ida_mat_f))
                if with_depthmap:
                    new_depthmaps.append(np.array(depthmap))

            else:
                resize, resize_dims, crop, flip, rotate = self._sample_augmentation(imgs[i])
                img = Image.fromarray(np.uint8(imgs[i]))
                if with_depthmap:
                    depthmap_im = Image.fromarray(results['depthmap'][i])
                else:
                    depthmap_im = None
                img, ida_mat, depthmap = self._img_transform(
                    img,
                    resize=resize,
                    resize_dims=resize_dims,
                    crop=crop,
                    flip=flip,
                    rotate=rotate,
                    depthmap=depthmap_im,
                )
                img = np.array(img).astype(np.float32)
                if 'gt_bboxes' in results.keys() and len(results['gt_bboxes']) > 0:
                    gt_bboxes = results['gt_bboxes'][i]
                    centers2d = results['centers2d'][i]
                    gt_labels = results['gt_labels'][i]
                    depths = results['depths'][i]
                    if len(gt_bboxes) != 0:
                        gt_bboxes, centers2d, gt_labels, depths = self._bboxes_transform(
                            img,
                            gt_bboxes,
                            centers2d,
                            gt_labels,
                            depths,
                            resize=resize,
                            crop=crop,
                            flip=flip,
                        )
                    if len(gt_bboxes) != 0:
                        gt_bboxes, centers2d, gt_labels, depths = self._filter_invisible(
                            img,
                            gt_bboxes,
                            centers2d,
                            gt_labels,
                            depths
                        )

                    new_gt_bboxes.append(gt_bboxes)
                    new_centers2d.append(centers2d)
                    new_gt_labels.append(gt_labels)
                    new_depths.append(depths)

                new_imgs.append(img)
                results['intrinsics'][i][:3, :3] = ida_mat @ results['intrinsics'][i][:3, :3]
                ida_mats.append(np.array(ida_mat))
                if with_depthmap:
                    new_depthmaps.append(np.array(depthmap))

        results['gt_bboxes'] = new_gt_bboxes
        results['centers2d'] = new_centers2d
        results['gt_labels'] = new_gt_labels
        results['depths'] = new_depths
        results['img'] = new_imgs
        results['cam2img'] = results['intrinsics']
        results['lidar2img'] = [results['intrinsics'][i] @ results['extrinsics'][i] for i in
                                range(len(results['extrinsics']))]

        results['img_shape'] = [img.shape for img in new_imgs]
        results['pad_shape'] = [img.shape for img in new_imgs]

        results['ida_mat'] = ida_mats  # shape N * (3, 3)

        if with_depthmap:
            results['depthmap'] = new_depthmaps

        return results

    def _bboxes_transform(self, img, bboxes, centers2d, gt_labels, depths, resize, crop, flip=False):
        assert len(bboxes) == len(centers2d) == len(gt_labels) == len(depths)

        fH, fW = img.shape[:2]
        bboxes = bboxes * resize
        bboxes[:, 0] = bboxes[:, 0] - crop[0]
        bboxes[:, 1] = bboxes[:, 1] - crop[1]
        bboxes[:, 2] = bboxes[:, 2] - crop[0]
        bboxes[:, 3] = bboxes[:, 3] - crop[1]
        bboxes[:, 0] = np.clip(bboxes[:, 0], 0, fW)
        bboxes[:, 2] = np.clip(bboxes[:, 2], 0, fW)
        bboxes[:, 1] = np.clip(bboxes[:, 1], 0, fH)
        bboxes[:, 3] = np.clip(bboxes[:, 3], 0, fH)
        keep = ((bboxes[:, 2] - bboxes[:, 0]) >= self.min_size) & ((bboxes[:, 3] - bboxes[:, 1]) >= self.min_size)

        if flip:
            x0 = bboxes[:, 0].copy()
            x1 = bboxes[:, 2].copy()
            bboxes[:, 2] = fW - x0
            bboxes[:, 0] = fW - x1
        # normalize
        bboxes = bboxes[keep]

        centers2d = centers2d * resize
        centers2d[:, 0] = centers2d[:, 0] - crop[0]
        centers2d[:, 1] = centers2d[:, 1] - crop[1]
        centers2d[:, 0] = np.clip(centers2d[:, 0], 0, fW)
        centers2d[:, 1] = np.clip(centers2d[:, 1], 0, fH)
        if flip:
            centers2d[:, 0] = fW - centers2d[:, 0]
        # normalize

        centers2d = centers2d[keep]
        gt_labels = gt_labels[keep]
        depths = depths[keep]

        return bboxes, centers2d, gt_labels, depths

    def offline_2d_transform(self, img, bboxes, resize, crop, flip=False):
        fH, fW = img.shape[:2]
        bboxes = np.array(bboxes).reshape(-1, 6)
        bboxes[..., :4] = bboxes[..., :4] * resize
        bboxes[:, 0] = bboxes[:, 0] - crop[0]
        bboxes[:, 1] = bboxes[:, 1] - crop[1]
        bboxes[:, 2] = bboxes[:, 2] - crop[0]
        bboxes[:, 3] = bboxes[:, 3] - crop[1]
        bboxes[:, 0] = np.clip(bboxes[:, 0], 0, fW)
        bboxes[:, 2] = np.clip(bboxes[:, 2], 0, fW)
        bboxes[:, 1] = np.clip(bboxes[:, 1], 0, fH)
        bboxes[:, 3] = np.clip(bboxes[:, 3], 0, fH)
        if flip:
            x0 = bboxes[:, 0].copy()
            x1 = bboxes[:, 2].copy()
            bboxes[:, 2] = fW - x0
            bboxes[:, 0] = fW - x1
        return bboxes

    def _filter_invisible(self, img, bboxes, centers2d, gt_labels, depths):
        assert len(bboxes) == len(centers2d) == len(gt_labels) == len(depths)

        fH, fW = img.shape[:2]
        indices_maps = np.zeros((fH, fW))
        tmp_bboxes = np.zeros_like(bboxes)
        tmp_bboxes[:, :2] = np.ceil(bboxes[:, :2])
        tmp_bboxes[:, 2:] = np.floor(bboxes[:, 2:])
        tmp_bboxes = tmp_bboxes.astype(np.int64)
        sort_idx = np.argsort(-depths, axis=0, kind='stable')
        tmp_bboxes = tmp_bboxes[sort_idx]
        bboxes = bboxes[sort_idx]
        depths = depths[sort_idx]
        centers2d = centers2d[sort_idx]
        gt_labels = gt_labels[sort_idx]
        for i in range(bboxes.shape[0]):
            u1, v1, u2, v2 = tmp_bboxes[i]
            indices_maps[v1:v2, u1:u2] = i
        indices_res = np.unique(indices_maps).astype(np.int64)
        bboxes = bboxes[indices_res]
        depths = depths[indices_res]
        centers2d = centers2d[indices_res]
        gt_labels = gt_labels[indices_res]

        return bboxes, centers2d, gt_labels, depths

    def _get_rot(self, h):
        return torch.Tensor(
            [
                [np.cos(h), np.sin(h)],
                [-np.sin(h), np.cos(h)],
            ]
        )

    def _img_transform(self, img, resize, resize_dims, crop, flip=False, rotate=0, depthmap=None):
        ida_rot = torch.eye(2)
        ida_tran = torch.zeros(2)
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        # adjust depthmap (int32)
        if depthmap is not None:
            depthmap = depthmap.resize(resize_dims, resample=Image.NEAREST)
            depthmap = depthmap.crop(crop)
            if flip:
                depthmap = depthmap.transpose(method=Image.FLIP_LEFT_RIGHT)
            depthmap = depthmap.rotate(rotate, resample=Image.NEAREST)

        # post-homography transformation
        ida_rot *= resize
        ida_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            ida_rot = A.matmul(ida_rot)
            ida_tran = A.matmul(ida_tran) + b
        A = self._get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        ida_rot = A.matmul(ida_rot)
        ida_tran = A.matmul(ida_tran) + b
        ida_mat = torch.eye(3)
        ida_mat[:2, :2] = ida_rot
        ida_mat[:2, 2] = ida_tran
        return img, ida_mat, depthmap

    def _sample_augmentation(self, img):
        # H, W = img.shape[:2]
        # fH, fW = self.data_aug_conf["final_dim"]
        # resize = np.random.uniform(*self.data_aug_conf["resize_lim"])
        # resize_dims = (int(W * resize), int(H * resize))
        # newW, newH = resize_dims
        # crop_h = int((1 - np.random.uniform(*self.data_aug_conf["bot_pct_lim"])) * newH) - fH
        # crop_w = int(np.random.uniform(0, max(0, newW - fW)))
        # crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        # flip = False
        # if self.data_aug_conf["rand_flip"] and np.random.choice([0, 1]):
        #     flip = True
        # rotate = np.random.uniform(*self.data_aug_conf["rot_lim"])

        H, W = img.shape[:2]
        fH, fW = self.data_aug_conf["final_dim"]
        if self.training:

            resize = np.random.uniform(*self.data_aug_conf["resize_lim"])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_aug_conf["bot_pct_lim"])) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf["rand_flip"] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf["rot_lim"])
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_aug_conf["bot_pct_lim"])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def _sample_augmentation_f(self, img):
        H, W = img.shape[:2]
        fH, fW = W, H

        resize = np.round(((H + 50) / W), 2)
        resize_dims = (int(W * resize), int(H * resize))
        newW, newH = resize_dims
        crop_h = int((newH - fH) / 2)
        crop_w = int((newW - fW) / 2)
        crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        return resize, resize_dims, crop


@PIPELINES.register_module()
class AV2PadMultiViewImage():
    """Pad the multi-view image.
    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",
    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value, 0 by default.
    """

    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        assert size is not None or size_divisor is not None
        assert size_divisor is None or size is None

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        if self.size == 'same2max':
            max_shape = max([img.shape for img in results['img']])[:2]
            padded_img = [mmcv.impad(img, shape=max_shape, pad_val=self.pad_val) for img in results['img']]
            with_depthmap = ('depthmap' in results.keys())
            if with_depthmap:
                results['depthmap'] = [mmcv.impad(depthmap, shape=max_shape, pad_val=self.pad_val) for depthmap in
                                       results['depthmap']]
        elif self.size is not None:
            padded_img = [mmcv.impad(img, shape=self.size, pad_val=self.pad_val) for img in results['img']]
        elif self.size_divisor is not None:
            padded_img = [
                mmcv.impad_to_multiple(img, self.size_divisor, pad_val=self.pad_val) for img in results['img']
            ]

        results['ori_shape'] = [img.shape for img in results['img']]
        results['img'] = padded_img
        results['img_shape'] = [img.shape for img in padded_img]
        results['pad_shape'] = [img.shape for img in padded_img]
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        self._pad_img(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_val={self.pad_val})'
        return repr_str


@PIPELINES.register_module()
class ResizeCropFlipRotImage():
    def __init__(self, data_aug_conf=None, with_2d=True, filter_invisible=True, training=True):
        self.data_aug_conf = data_aug_conf
        self.training = training
        self.min_size = 2.0
        self.with_2d = with_2d
        self.filter_invisible = filter_invisible

        self.cached_scene_augs = dict()
        self.scene_next_token = dict()

    def __call__(self, results):

        imgs = results['img']
        N = len(imgs)
        new_imgs = []
        new_gt_bboxes = []
        new_centers2d = []
        new_gt_labels = []
        new_depths = []
        new_proposals = []
        new_gt_inds = []

        assert self.data_aug_conf['rot_lim'] == (0.0, 0.0), "Rotation is not currently supported"
        resize, resize_dims, crop, flip, rotate = self._sample_augmentation()

        for i in range(N):
            img = Image.fromarray(np.uint8(imgs[i]))
            img, ida_mat = self._img_transform(
                img,
                resize=resize,
                resize_dims=resize_dims,
                crop=crop,
                flip=flip,
                rotate=rotate,
            )
            if self.with_2d: # sync_2d bbox labels
                if 'gt_bboxes' in results:
                    gt_bboxes = results['gt_bboxes'][i]
                    centers2d = results['centers2d'][i]
                    gt_labels = results['gt_labels'][i]
                    depths = results['depths'][i]
                    gt_inds = None if 'instance_inds_2d' not in results else results['instance_inds_2d'][i]
                    if len(gt_bboxes) != 0:
                        gt_bboxes, centers2d, gt_labels, depths, gt_inds = self._bboxes_transform(
                            gt_bboxes,
                            centers2d,
                            gt_labels,
                            depths,
                            resize=resize,
                            crop=crop,
                            flip=flip,
                            gt_inds=gt_inds,
                        )
                    if len(gt_bboxes) != 0 and self.filter_invisible:
                        gt_bboxes, centers2d, gt_labels, depths, gt_inds = self._filter_invisible(gt_bboxes, centers2d, gt_labels, depths, gt_inds=gt_inds)

                    new_gt_bboxes.append(gt_bboxes)
                    new_centers2d.append(centers2d)
                    new_gt_labels.append(gt_labels)
                    new_depths.append(depths)
                    new_gt_inds.append(gt_inds)

                if 'proposals' in results:
                    proposals = results['proposals'][i]
                    if len(proposals) > 0:
                        proposals = self._proposals_transform(proposals, resize, crop, flip)
                        if self.filter_invisible:
                            fH, fW = self.data_aug_conf["final_dim"]
                            proposal_boxes = proposals[..., :4]
                            xy_max = np.minimum(proposal_boxes[..., 2:4], [fW-1, fH-1])
                            xy_min = np.maximum(proposal_boxes[..., 0:2], [0, 0])
                            wh = np.maximum(xy_max - xy_min, 0)
                            keep = (wh >= self.min_size).all(-1)
                            proposals = proposals[keep]
                    new_proposals.append(proposals)

            new_imgs.append(np.array(img).astype(np.float32))
            results['intrinsics'][i][:3, :3] = ida_mat @ results['intrinsics'][i][:3, :3]
        results['gt_bboxes'] = new_gt_bboxes
        results['centers2d'] = new_centers2d
        results['gt_labels'] = new_gt_labels
        results['depths'] = new_depths
        results['img'] = new_imgs
        results['lidar2img'] = [results['intrinsics'][i] @ results['extrinsics'][i] for i in range(len(results['extrinsics']))]
        if 'proposals' in results:
            results['proposals'] = new_proposals
        if 'instance_inds_2d' in results:
            results['instance_inds_2d'] = new_gt_inds

        return results

    def _proposals_transform(self, proposals, resize, crop, flip):
        bboxes = proposals[:, :4]
        fH, fW = self.data_aug_conf["final_dim"]
        bboxes = bboxes * resize
        bboxes[:, 0] = bboxes[:, 0] - crop[0]
        bboxes[:, 1] = bboxes[:, 1] - crop[1]
        bboxes[:, 2] = bboxes[:, 2] - crop[0]
        bboxes[:, 3] = bboxes[:, 3] - crop[1]
        bboxes[:, 0] = np.clip(bboxes[:, 0], 0, fW)
        bboxes[:, 2] = np.clip(bboxes[:, 2], 0, fW)
        bboxes[:, 1] = np.clip(bboxes[:, 1], 0, fH)
        bboxes[:, 3] = np.clip(bboxes[:, 3], 0, fH)
        keep = ((bboxes[:, 2] - bboxes[:, 0]) >= self.min_size) & ((bboxes[:, 3] - bboxes[:, 1]) >= self.min_size)

        if flip:
            x0 = bboxes[:, 0].copy()
            x1 = bboxes[:, 2].copy()
            bboxes[:, 2] = fW - x0
            bboxes[:, 0] = fW - x1

        proposals = np.concatenate([bboxes, proposals[:, 4:]], axis=1)
        proposals = proposals[keep]
        return proposals

    def _bboxes_transform(self, bboxes, centers2d, gt_labels, depths, resize, crop, flip, gt_inds=None):
        assert len(bboxes) == len(centers2d) == len(gt_labels) == len(depths)
        fH, fW = self.data_aug_conf["final_dim"]
        bboxes = bboxes * resize
        bboxes[:, 0] = bboxes[:, 0] - crop[0]
        bboxes[:, 1] = bboxes[:, 1] - crop[1]
        bboxes[:, 2] = bboxes[:, 2] - crop[0]
        bboxes[:, 3] = bboxes[:, 3] - crop[1]
        bboxes[:, 0] = np.clip(bboxes[:, 0], 0, fW)
        bboxes[:, 2] = np.clip(bboxes[:, 2], 0, fW)
        bboxes[:, 1] = np.clip(bboxes[:, 1], 0, fH) 
        bboxes[:, 3] = np.clip(bboxes[:, 3], 0, fH)
        keep = ((bboxes[:, 2] - bboxes[:, 0]) >= self.min_size) & ((bboxes[:, 3] - bboxes[:, 1]) >= self.min_size)


        if flip:
            x0 = bboxes[:, 0].copy()
            x1 = bboxes[:, 2].copy()
            bboxes[:, 2] = fW - x0
            bboxes[:, 0] = fW - x1
        bboxes = bboxes[keep]

        centers2d  = centers2d * resize
        centers2d[:, 0] = centers2d[:, 0] - crop[0]
        centers2d[:, 1] = centers2d[:, 1] - crop[1]
        centers2d[:, 0] = np.clip(centers2d[:, 0], 0, fW)
        centers2d[:, 1] = np.clip(centers2d[:, 1], 0, fH) 
        if flip:
            centers2d[:, 0] = fW - centers2d[:, 0]

        centers2d = centers2d[keep]
        gt_labels = gt_labels[keep]
        depths = depths[keep]
        if gt_inds is not None:
            gt_inds = gt_inds[keep]

        return bboxes, centers2d, gt_labels, depths, gt_inds

    def _filter_invisible(self, bboxes, centers2d, gt_labels, depths, gt_inds=None):
        # filter invisible 2d bboxes
        assert len(bboxes) == len(centers2d) == len(gt_labels) == len(depths)
        if gt_inds is not None:
            assert len(gt_labels) == len(gt_inds)
        fH, fW = self.data_aug_conf["final_dim"]
        indices_maps = np.zeros((fH,fW))
        tmp_bboxes = np.zeros_like(bboxes)
        tmp_bboxes[:, :2] = np.ceil(bboxes[:, :2])
        tmp_bboxes[:, 2:] = np.floor(bboxes[:, 2:])
        tmp_bboxes = tmp_bboxes.astype(np.int64)
        sort_idx = np.argsort(-depths, axis=0, kind='stable')
        tmp_bboxes = tmp_bboxes[sort_idx]
        bboxes = bboxes[sort_idx]
        depths = depths[sort_idx]
        centers2d = centers2d[sort_idx]
        gt_labels = gt_labels[sort_idx]
        if gt_inds is not None:
            gt_inds = gt_inds[sort_idx]
        for i in range(bboxes.shape[0]):
            u1, v1, u2, v2 = tmp_bboxes[i]
            indices_maps[v1:v2, u1:u2] = i
        indices_res = np.unique(indices_maps).astype(np.int64)
        bboxes = bboxes[indices_res]
        depths = depths[indices_res]
        centers2d = centers2d[indices_res]
        gt_labels = gt_labels[indices_res]
        if gt_inds is not None:
            gt_inds = gt_inds[indices_res]

        return bboxes, centers2d, gt_labels, depths, gt_inds

    def _get_rot(self, h):
        return torch.Tensor(
            [
                [np.cos(h), np.sin(h)],
                [-np.sin(h), np.cos(h)],
            ]
        )

    def _img_transform(self, img, resize, resize_dims, crop, flip, rotate):
        ida_rot = torch.eye(2)
        ida_tran = torch.zeros(2)
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        # post-homography transformation
        ida_rot *= resize
        ida_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            ida_rot = A.matmul(ida_rot)
            ida_tran = A.matmul(ida_tran) + b
        A = self._get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        ida_rot = A.matmul(ida_rot)
        ida_tran = A.matmul(ida_tran) + b
        ida_mat = torch.eye(3)
        ida_mat[:2, :2] = ida_rot
        ida_mat[:2, 2] = ida_tran
        return img, ida_mat

    def _sample_augmentation(self):
        H, W = self.data_aug_conf["H"], self.data_aug_conf["W"]
        fH, fW = self.data_aug_conf["final_dim"]
        if self.training:
            resize = np.random.uniform(*self.data_aug_conf["resize_lim"])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_aug_conf["bot_pct_lim"])) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf["rand_flip"] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf["rot_lim"])
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_aug_conf["bot_pct_lim"])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

@PIPELINES.register_module()
class GlobalRotScaleTransImage():
    def __init__(
        self,
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0],
        reverse_angle=False,
        training=True,
    ):

        self.rot_range = rot_range
        self.scale_ratio_range = scale_ratio_range
        self.translation_std = translation_std

        self.reverse_angle = reverse_angle
        self.training = training

    def __call__(self, results):
        # random rotate
        translation_std = np.array(self.translation_std, dtype=np.float32)

        rot_angle = np.random.uniform(*self.rot_range)
        scale_ratio = np.random.uniform(*self.scale_ratio_range)
        trans = np.random.normal(scale=translation_std, size=3).T

        self._rotate_bev_along_z(results, rot_angle)
        if self.reverse_angle:
            rot_angle = rot_angle * -1
        results["gt_bboxes_3d"].rotate(
            np.array(rot_angle)
        )  

        # random scale
        self._scale_xyz(results, scale_ratio)
        results["gt_bboxes_3d"].scale(scale_ratio)

        #random translate
        self._trans_xyz(results, trans)
        results["gt_bboxes_3d"].translate(trans)

        return results

    def _trans_xyz(self, results, trans):
        trans_mat = torch.eye(4, 4)
        trans_mat[:3, -1] = torch.from_numpy(trans).reshape(1, 3)
        trans_mat_inv = torch.inverse(trans_mat)
        num_view = len(results["lidar2img"])
        results['ego_pose'] = (torch.tensor(results["ego_pose"]).float() @ trans_mat_inv).numpy()
        results['ego_pose_inv'] = (trans_mat.float() @ torch.tensor(results["ego_pose_inv"])).numpy()

        for view in range(num_view):
            results["lidar2img"][view] = (torch.tensor(results["lidar2img"][view]).float() @ trans_mat_inv).numpy()
            results["extrinsics"][view] = (torch.tensor(results["extrinsics"][view]).float() @ trans_mat_inv).numpy()

    def _rotate_bev_along_z(self, results, angle):
        rot_cos = torch.cos(torch.tensor(angle))
        rot_sin = torch.sin(torch.tensor(angle))

        rot_mat = torch.tensor([[rot_cos, rot_sin, 0, 0], [-rot_sin, rot_cos, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        rot_mat_inv = torch.inverse(rot_mat)

        results['ego_pose'] = (torch.tensor(results["ego_pose"]).float() @ rot_mat_inv).numpy()
        results['ego_pose_inv'] = (rot_mat.float() @ torch.tensor(results["ego_pose_inv"])).numpy()
        num_view = len(results["lidar2img"])
        for view in range(num_view):
            results["lidar2img"][view] = (torch.tensor(results["lidar2img"][view]).float() @ rot_mat_inv).numpy()
            results["extrinsics"][view] = (torch.tensor(results["extrinsics"][view]).float() @ rot_mat_inv).numpy()

    def _scale_xyz(self, results, scale_ratio):
        scale_mat = torch.tensor(
            [
                [scale_ratio, 0, 0, 0],
                [0, scale_ratio, 0, 0],
                [0, 0, scale_ratio, 0],
                [0, 0, 0, 1],
            ]
        )

        scale_mat_inv = torch.inverse(scale_mat)

        results['ego_pose'] = (torch.tensor(results["ego_pose"]).float() @ scale_mat_inv).numpy()
        results['ego_pose_inv'] = (scale_mat @ torch.tensor(results["ego_pose_inv"]).float()).numpy()

        num_view = len(results["lidar2img"])
        for view in range(num_view):
            results["lidar2img"][view] = (torch.tensor(results["lidar2img"][view]).float() @ scale_mat_inv).numpy()
            results["extrinsics"][view] = (torch.tensor(results["extrinsics"][view]).float() @ scale_mat_inv).numpy()


@PIPELINES.register_module()
class BEVGlobalRotScaleTrans(GlobalRotScaleTransImage):
    def __call__(self, results):
        # random rotate
        translation_std = np.array(self.translation_std, dtype=np.float32)

        rot_angle = np.random.uniform(*self.rot_range)
        scale_ratio = np.random.uniform(*self.scale_ratio_range)
        trans = np.random.normal(scale=translation_std, size=3).T

        points = results['points']

        self._rotate_bev_along_z(results, rot_angle)
        if self.reverse_angle:
            rot_angle = rot_angle * -1
        points, _ = results["gt_bboxes_3d"].rotate(
            np.array(rot_angle), points
        )

        # random scale
        self._scale_xyz(results, scale_ratio)
        results["gt_bboxes_3d"].scale(scale_ratio)
        points.scale(scale_ratio)

        #random translate
        self._trans_xyz(results, trans)
        results["gt_bboxes_3d"].translate(trans)
        points.translate(trans)
        results['points'] = points

        return results


@PIPELINES.register_module()
class BEVRandomFlip3D:
    """Compared with `RandomFlip3D`, this class directly records the lidar
    augmentation matrix in the `data`."""

    def __call__(self, results):
        flip_horizontal = np.random.choice([0, 1])
        flip_vertical = np.random.choice([0, 1])

        rotation = np.eye(3)
        if flip_horizontal:
            rotation = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]) @ rotation
            if 'points' in results:
                results['points'].flip('horizontal')
            if 'gt_bboxes_3d' in results:
                results['gt_bboxes_3d'].flip('horizontal')

        if flip_vertical:
            rotation = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]) @ rotation
            if 'points' in results:
                results['points'].flip('vertical')
            if 'gt_bboxes_3d' in results:
                results['gt_bboxes_3d'].flip('vertical')

        rot_mat = np.eye(4)
        rot_mat[:3, :3] = rotation
        rot_mat = torch.from_numpy(rot_mat).float()
        rot_mat_inv = torch.inverse(rot_mat)
        results['ego_pose'] = (torch.tensor(results["ego_pose"]).float() @ rot_mat_inv).numpy()
        results['ego_pose_inv'] = (rot_mat.float() @ torch.tensor(results["ego_pose_inv"])).numpy()
        num_view = len(results["lidar2img"])
        for view in range(num_view):
            results["lidar2img"][view] = (torch.tensor(results["lidar2img"][view]).float() @ rot_mat_inv).numpy()
            results["extrinsics"][view] = (torch.tensor(results["extrinsics"][view]).float() @ rot_mat_inv).numpy()
        return results