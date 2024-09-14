_base_ = [
    '../_base_/default_runtime.py',
]
plugin = True
plugin_dir = [
    'projects/mmdet3d_plugin/',
    'projects/fsdv2/'
]

class_names = ['ARTICULATED_BUS', 'BICYCLE', 'BICYCLIST', 'BOLLARD', 'BOX_TRUCK', 'BUS',
               'CONSTRUCTION_BARREL', 'CONSTRUCTION_CONE', 'DOG', 'LARGE_VEHICLE',
               'MESSAGE_BOARD_TRAILER', 'MOBILE_PEDESTRIAN_CROSSING_SIGN', 'MOTORCYCLE',
               'MOTORCYCLIST', 'PEDESTRIAN', 'REGULAR_VEHICLE', 'SCHOOL_BUS', 'SIGN',
               'STOP_SIGN', 'STROLLER', 'TRUCK', 'TRUCK_CAB', 'VEHICULAR_TRAILER',
               'WHEELCHAIR', 'WHEELED_DEVICE','WHEELED_RIDER']

# FSDv2 setting
point_cloud_range = [-204.8, -204.8, -3.2, 204.8, 204.8, 3.2]
seg_voxel_size = (0.2, 0.2, 0.2)
virtual_voxel_size=(0.4, 0.4, 0.4)
sparse_shape=[32, 2048, 2048]
target_sparse_shape = [16, 1024, 1024]
fsd_class_names = \
['Regular_vehicle',
 'Pedestrian',
 'Bicyclist',
 'Motorcyclist',
 'Wheeled_rider',
 'Bollard',
 'Construction_cone',
 'Sign',
 'Construction_barrel',
 'Stop_sign',
 'Mobile_pedestrian_crossing_sign',
 'Large_vehicle',
 'Bus',
 'Box_truck',
 'Truck',
 'Vehicular_trailer',
 'Truck_cab',
 'School_bus',
 'Articulated_bus',
 'Message_board_trailer',
 'Bicycle',
 'Motorcycle',
 'Wheeled_device',
 'Wheelchair',
 'Stroller',
 'Dog']
group1 = fsd_class_names[:1]
group2 = fsd_class_names[1:5]
group3 = fsd_class_names[5:11]
group4 = fsd_class_names[11:20]
group5 = fsd_class_names[20:25]
group6 = fsd_class_names[25:]
assert len(group6) == 1
sample_group_1 = {k:1 for k in group1}
sample_group_2 = {k:2 for k in group2}
sample_group_3 = {k:2 for k in group3}
sample_group_4 = {k:1 for k in group4}
sample_group_5 = {k:2 for k in group5}
sample_group_6 = {k:2 for k in group6}
#merge all groups
sample_groups = {**sample_group_1, **sample_group_2, **sample_group_3, **sample_group_4, **sample_group_5, **sample_group_6}
sample_groups.update({'Wheelchair':0, 'Dog':0, 'Message_board_trailer':0})

seg_score_thresh = [0.4, 0.25, 0.25, 0.25, 0.25, 0.25]
group_lens = [len(group1), len(group2), len(group3), len(group4), len(group5), len(group6)]

# training hyperparameter
num_gpus = 8
batch_size = 1
num_iters_per_epoch = 110071 // (num_gpus * batch_size)
num_epochs = 6

queue_length = 1
num_frame_losses = 1

pts_ckpt = 'weights/fsdv2-argo-converted.pth'
img_ckpt = 'weights/mask_rcnn_r50_fpn_1x_nuim_20201008_195238-e99f5182.pth'

roi_size = 7
roi_strides = [4, 8, 16, 32, 64]
model = dict(
    type='MV2DFusion',
    dataset='argov2',
    num_frame_head_grads=num_frame_losses,
    num_frame_backbone_grads=num_frame_losses,
    num_frame_losses=num_frame_losses,
    position_level=2,
    use_grid_mask=False,

    loss_weight_3d=0.1,
    loss_weight_pts=1.,
    gt_mono_loss=True,

    img_backbone=dict(
        init_cfg=dict(
            type='Pretrained', checkpoint=img_ckpt,
            prefix='backbone.', map_location='cpu'),
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        with_cp=True,
        style='pytorch'),
    img_neck=dict(
        init_cfg=dict(
            type='Pretrained', checkpoint=img_ckpt,
            prefix='neck.', map_location='cpu'),
        norm_cfg=dict(type='SyncBN'),
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    img_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', output_size=roi_size, sampling_ratio=-1),
        featmap_strides=roi_strides[:-1],
        out_channels=256, ),
    # faster rcnn
    img_roi_head=dict(
        type='TwoStageDetectorWrapper',
        init_cfg=dict(
            type='Pretrained', checkpoint=img_ckpt,
            map_location='cpu'),
        rpn_head=dict(
            type='RPNHead',
            in_channels=256,
            feat_channels=256,
            anchor_generator=dict(
                type='AnchorGenerator',
                scales=[8],
                ratios=[0.5, 1.0, 2.0],
                strides=roi_strides),
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[.0, .0, .0, .0],
                target_stds=[1.0, 1.0, 1.0, 1.0]),
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0),
        ),
        roi_head=dict(
            type='StandardRoIHead',
            bbox_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
                out_channels=256,
                featmap_strides=roi_strides[:-1]),
            bbox_head=dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=len(class_names),
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=False,
                loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=1.0),
                reg_decoded_bbox=True,
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0)
            ),
        ),
        train_cfg=dict(
            rpn=dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.3,
                    min_pos_iou=0.3,
                    match_low_quality=True,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=256,
                    pos_fraction=0.5,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=False),
                allowed_border=-1,
                pos_weight=-1,
                debug=False),
            rpn_proposal=dict(
                nms_pre=2000,
                max_per_img=1000,
                nms=dict(type='nms', iou_threshold=0.7),
                min_bbox_size=0),
            rcnn=dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=True,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False)),
        test_cfg=dict(
            min_bbox_size=4,
            rpn=dict(
                nms_pre=1000,
                max_per_img=1000,
                nms=dict(type='nms', iou_threshold=0.7),
                min_bbox_size=0),
            rcnn=dict(
                score_thr=0.05,
                nms=dict(type='nms', iou_threshold=0.6, class_agnostic=True),
                max_per_img=60,)),
    ),
    img_query_generator=dict(
        type='ImageDistributionQueryGenerator',
        num_classes=len(class_names),
        code_size=8,
        prob_bin=50,
        depth_range=[0.1, 240],
        gt_guided=False,
        gt_guided_loss=1.,
        with_cls=True,
        with_size=True,

        with_avg_pool=True,
        num_shared_convs=1,
        num_shared_fcs=1,
        in_channels=256,
        fc_out_channels=1024,
        roi_feat_size=roi_size,
        extra_encoding=dict(
            num_layers=2,
            feat_channels=[512, 256],
            features=[dict(type='intrinsic', in_channels=16, )]
        ),
    ),
    pts_backbone=dict(
        init_cfg=dict(
            type='Pretrained', checkpoint=pts_ckpt, prefix='pts_backbone.', map_location='cpu'),
        type='SingleStageFSDV2',
        freeze=False,
        norm_eval=True,
        segmentor=dict(
            type='VoteSegmentor',
            voxel_layer=dict(
                voxel_size=seg_voxel_size,
                max_num_points=-1,
                point_cloud_range=point_cloud_range,
                max_voxels=(-1, -1)
            ),
            voxel_encoder=dict(
                type='DynamicScatterVFE',
                in_channels=4,
                feat_channels=[64, 64],
                with_distance=False,
                voxel_size=seg_voxel_size,
                with_cluster_center=True,
                with_voxel_center=True,
                point_cloud_range=point_cloud_range,
                norm_cfg=dict(type='MyBN1d', eps=1e-3, momentum=0.01),
            ),
            middle_encoder=dict(
                type='PseudoMiddleEncoderForSpconvFSD',
            ),
            backbone=dict(
                type='SimpleSparseUNet',
                in_channels=64,
                sparse_shape=sparse_shape,
                order=('conv', 'norm', 'act'),
                norm_cfg=dict(type='MyBN1d', eps=1e-3, momentum=0.01),
                base_channels=64,
                output_channels=128, # dummy
                encoder_channels=((128, ), (128, 128, ), (128, 128, ), (128, 128, 128), (256, 256, 256), (256, 256, 256)),
                encoder_paddings=((1, ), (1, 1, ), (1, 1, ), (1, 1, 1), (1, 1, 1), (1, 1, 1)),
                decoder_channels=((256, 256, 256), (256, 256, 128), (128, 128, 128), (128, 128, 128), (128, 128, 128), (128, 128, 128)),
                decoder_paddings=((1, 1), (1, 0), (1, 0), (0, 0), (0, 1), (1, 1)), # decoder paddings seem useless in SubMConv
                return_multiscale_features=True,
            ),
            decode_neck=dict(
                type='Voxel2PointScatterNeck',
                voxel_size=seg_voxel_size,
                point_cloud_range=point_cloud_range,
            ),
            segmentation_head=dict(
                type='VoteSegHead',
                in_channel=67 + 64,
                hidden_dims=[128, 128],
                num_classes=len(class_names),
                dropout_ratio=0.0,
                conv_cfg=dict(type='Conv1d'),
                norm_cfg=dict(type='MyBN1d'),
                act_cfg=dict(type='ReLU'),
                loss_decode=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    class_weight=[1.0, ] * len(class_names) + [0.1,],
                    loss_weight=3.0),
                loss_vote=dict(
                    type='L1Loss',
                    loss_weight=1.0),
            ),
            train_cfg=dict(
                point_loss=True,
                score_thresh=seg_score_thresh, # for training log
                class_names=fsd_class_names, # for training log
                group_names=[group1, group2, group3, group4, group5, group6],
                group_lens=group_lens,
            ),
        ),
        virtual_point_projector=dict(
            in_channels=98 + 64,
            hidden_dims=[64, 64],
            norm_cfg=dict(type='MyBN1d'),
            ori_in_channels=67 + 64,
            ori_hidden_dims=[64, 64],
        ),
        multiscale_cfg=dict(
            multiscale_levels=[0, 1, 2],
            projector_hiddens=[[256, 128], [128, 128], [128, 128]],
            fusion_mode='avg',
            target_sparse_shape=target_sparse_shape,
            norm_cfg=dict(type='MyBN1d'),
        ),
        voxel_encoder=dict(
            type='DynamicScatterVFE',
            in_channels=67,
            feat_channels=[64, 128],
            voxel_size=virtual_voxel_size,
            with_cluster_center=True,
            with_voxel_center=True,
            point_cloud_range=point_cloud_range,
            norm_cfg=dict(type='MyBN1d', eps=1e-3, momentum=0.01),
            unique_once=True,
        ),
        backbone=dict(
            type='VirtualVoxelMixer',
            in_channels=128,
            sparse_shape=target_sparse_shape,
            order=('conv', 'norm', 'act'),
            norm_cfg=dict(type='MyBN1d', eps=1e-3, momentum=0.01),
            base_channels=64,
            output_channels=128,
            encoder_channels=((64,), (64, 64,), (64, 64,),),
            encoder_paddings=((1,), (1, 1,), (1, 1,),),
            decoder_channels=((64, 64, 64), (64, 64, 64), (64, 64, 64)),
            decoder_paddings=((1, 1), (1, 1), (1, 1),),  # decoder paddings seem useless in SubMConv
        ),
        bbox_head=dict(
            type='FSDV2Head',
            as_rpn=False,
            num_classes=len(class_names),
            bbox_coder=dict(type='BasePointBBoxCoder', code_size=8),
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=1.0,
                alpha=0.25,
                loss_weight=4.0),
            loss_center=dict(type='SmoothL1Loss', loss_weight=0.25, beta=0.1),
            loss_size=dict(type='SmoothL1Loss', loss_weight=0.25, beta=0.1),
            loss_rot=dict(type='SmoothL1Loss', loss_weight=0.1, beta=0.1),
            in_channel=128,
            shared_mlp_dims=[256, 256],
            train_cfg=None,
            test_cfg=None,
            norm_cfg=dict(type='LN'),
            tasks=[
                dict(class_names=group1),
                dict(class_names=group2),
                dict(class_names=group3),
                dict(class_names=group4),
                dict(class_names=group5),
                dict(class_names=group6),
            ],
            class_names=fsd_class_names,
            common_attrs=dict(
                center=(3, 2, 128), dim=(3, 2, 128), rot=(2, 2, 128),
                # (out_dim, num_layers, hidden_dim)
            ),
            num_cls_layer=2,
            cls_hidden_dim=128,
            separate_head=dict(
                type='FSDSeparateHead',
                norm_cfg=dict(type='LN'),
                act='relu',
            ),
        ),
        train_cfg=dict(
            score_thresh=seg_score_thresh,
            sync_reg_avg_factor=True,
            batched_group_sample=True,
            offset_weight='max',
            class_names=fsd_class_names,
            group_names=[group1, group2, group3, group4, group5, group6],
            centroid_assign=True,
            disable_pretrain=True,
            disable_pretrain_topks=[300, ] * len(class_names),
        ),
        test_cfg=dict(
            score_thresh=seg_score_thresh,
            batched_group_sample=True,
            offset_weight='max',
            class_names=fsd_class_names,
            group_names=[group1, group2, group3, group4, group5, group6],
            use_rotate_nms=True,
            nms_pre=-1,
            nms_thr=0.15,
            score_thr=0.05,
            min_bbox_size=0,
            max_num=500,
            all_task_max_num=200,
        ),
    ),
    pts_query_generator=dict(
        type='PointCloudQueryGenerator',
        in_channels=128,
        hidden_channel=128,
        pts_use_cat=True,
    ),
    # fusion head
    fusion_bbox_head=dict(
        type='MV2DFusionHead',
        code_size=8,
        prob_bin=50,
        post_bev_nms_ops=[],
        num_classes=len(class_names),
        in_channels=256,
        num_query=300,
        memory_len=6 * 256,
        topk_proposals=256,
        num_propagated=256,
        with_ego_pos=True,
        match_with_velo=False,
        scalar=10,  ##noise groups
        noise_scale=1.0,
        dn_weight=1.0,  ##dn loss weight
        split=0.75,  ###positive rate
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        transformer=dict(
            type='MV2DFusionTransformer',
            decoder=dict(
                type='MV2DFusionTransformerDecoder',
                return_intermediate=True,
                num_layers=6,
                transformerlayers=dict(
                    type='MV2DFusionTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='MixedCrossAttention',
                            embed_dims=256,
                            num_groups=8,
                            num_levels=4,
                            num_cams=7,
                            dropout=0.1,
                            num_pts=13,
                            bias=2.,
                            attn_cfg=dict(
                                type='PETRMultiheadFlashAttention',
                                batch_first=False,
                                embed_dims=256,
                                num_heads=8,
                                dropout=0.1),),
                    ],
                    feedforward_channels=2048,
                    ffn_dropout=0.1,
                    with_cp=True,  ###use checkpoint to save memory
                    operation_order=(
                        'self_attn', 'norm',
                        'cross_attn', 'norm',
                        'ffn', 'norm')
                ),
            )),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=point_cloud_range,
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=None,
            num_classes=len(class_names)),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0), ),
    train_cfg=dict(fusion=dict(
        point_cloud_range=point_cloud_range,
        out_size_factor=4,
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0.0),  # Fake cost. This is just to make it compatible with DETR head.
            pc_range=point_cloud_range), )))

# data pipeline
file_client_args = dict(backend='disk')
collect_keys = ['lidar2img', 'intrinsics', 'extrinsics', 'timestamp', 'img_timestamp', 'ego_pose', 'ego_pose_inv']
input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)  # we use nuimages pretrain for 2D detector
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
ida_aug_conf = {
    "resize_lim": (0.65, 0.9),  # 0.75
    "final_dim": (1184, 1536),
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (0.0, 0.0),
    "rand_flip": True,
}

dataset_type = 'Argoverse2DatasetT'
data_root = './data/argo/converted/'

train_pipeline = [
    dict(
        type='AV2LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=file_client_args),
    dict(type='AV2LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_bbox=True,
        with_label=True, with_bbox_depth=True),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='AV2ResizeCropFlipRotImageV2', data_aug_conf=ida_aug_conf, training=True,),
    dict(type='BEVGlobalRotScaleTrans',
         rot_range=[-0.3925, 0.3925],
         translation_std=[0, 0, 0],
         scale_ratio_range=[0.95, 1.05],
         reverse_angle=True,
         training=True,
         ),
    dict(type='BEVRandomFlip3D'),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='AV2PadMultiViewImage', size='same2max'),
    dict(type='PETRFormatBundle3D', class_names=class_names,
         collect_keys=collect_keys + ['prev_exists']),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d', 'img', 'gt_bboxes', 'gt_labels', 'centers2d', 'depths',
                                 'prev_exists'] + collect_keys,
         meta_keys=(
             'filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'box_mode_3d', 'box_type_3d',
             'img_norm_cfg', 'scene_token', 'gt_bboxes_3d', 'gt_labels_3d'), )
]
test_pipeline = [
    dict(
        type='AV2LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=file_client_args),
    dict(type='AV2LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_bbox=True,
        with_label=True, with_bbox_depth=True),
    dict(type='AV2ResizeCropFlipRotImageV2', data_aug_conf=ida_aug_conf),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='AV2PadMultiViewImage', size='same2max'),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='PETRFormatBundle3D',
                collect_keys=collect_keys + ['prev_exists'],
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D',
                 keys=['points', 'img', 'prev_exists'] + collect_keys ,
                 meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'box_mode_3d',
                            'box_type_3d', 'img_norm_cfg', 'scene_token', 'gt_bboxes_3d', 'gt_labels_3d'))
        ])
]

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'av2_train_infos_mini.pkl',
        split='train',
        load_interval=1,
        num_frame_losses=num_frame_losses,
        seq_split_num=2,
        seq_mode=True,
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        collect_keys=collect_keys + ['img', 'prev_exists', 'img_metas'],
        queue_length=queue_length,
        test_mode=False,
        use_valid_flag=False,
        interval_test=True,
        box_type_3d='LiDAR'),
    val=dict(
        type=dataset_type,
        pipeline=test_pipeline,
        data_root=data_root,
        collect_keys=collect_keys + ['img', 'img_metas'],
        queue_length=queue_length,
        ann_file=data_root + 'av2_val_infos_mini.pkl',
        split='val',
        load_interval=1,
        classes=class_names,
        modality=input_modality,
        interval_test=True,),
    test=dict(
        type=dataset_type,
        pipeline=test_pipeline,
        data_root=data_root,
        collect_keys=collect_keys + ['img', 'img_metas'],
        queue_length=queue_length,
        ann_file=data_root + 'av2_val_infos_mini.pkl',
        split='val',
        load_interval=1,
        classes=class_names,
        modality=input_modality,
        interval_test=True,),
    shuffler_sampler=dict(type='InfiniteGroupEachSampleInBatchSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)

optimizer = dict(
    type='AdamW',
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
            'pts_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)

optimizer_config = dict(
    type='Fp16OptimizerHook', loss_scale='dynamic',
    grad_clip=dict(max_norm=35, norm_type=2))

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
)

custom_hooks = []

evaluation = dict(interval=num_epochs * num_iters_per_epoch, pipeline=test_pipeline)
checkpoint_config = dict(interval=num_iters_per_epoch // 4, save_last=True)
find_unused_parameters = False  #### when use checkpoint, find_unused_parameters must be False
runner = dict(
    type='IterBasedRunner', max_iters=num_epochs * num_iters_per_epoch)
load_from = None
resume_from = None

#                                     AP    ATE    ASE    AOE    CDS  RECALL
# ARTICULATED_BUS                  0.280  0.743  0.181  0.061  0.227   0.448
# BICYCLE                          0.641  0.322  0.261  0.305  0.530   0.722
# BICYCLIST                        0.672  0.301  0.260  0.199  0.566   0.754
# BOLLARD                          0.650  0.108  0.442  0.857  0.484   0.746
# BOX_TRUCK                        0.482  0.543  0.218  0.055  0.400   0.615
# BUS                              0.553  0.556  0.177  0.107  0.463   0.646
# CONSTRUCTION_BARREL              0.731  0.136  0.227  0.653  0.609   0.777
# CONSTRUCTION_CONE                0.712  0.120  0.416  0.839  0.536   0.785
# DOG                              0.314  0.422  0.475  0.889  0.212   0.515
# LARGE_VEHICLE                    0.086  0.693  0.305  0.322  0.065   0.305
# MESSAGE_BOARD_TRAILER            0.000  0.333  0.195  3.098  0.000   0.015
# MOBILE_PEDESTRIAN_CROSSING_SIGN  0.595  0.079  0.361  1.393  0.428   0.623
# MOTORCYCLE                       0.757  0.291  0.249  0.257  0.637   0.824
# MOTORCYCLIST                     0.726  0.300  0.273  0.152  0.612   0.811
# PEDESTRIAN                       0.781  0.235  0.231  0.488  0.650   0.837
# REGULAR_VEHICLE                  0.816  0.351  0.151  0.109  0.718   0.854
# SCHOOL_BUS                       0.570  0.568  0.193  0.112  0.473   0.656
# SIGN                             0.330  0.290  0.303  0.378  0.268   0.566
# STOP_SIGN                        0.579  0.168  0.379  0.147  0.480   0.657
# STROLLER                         0.409  0.242  0.312  0.235  0.340   0.595
# TRUCK                            0.269  0.514  0.145  0.105  0.230   0.510
# TRUCK_CAB                        0.345  0.722  0.242  0.076  0.273   0.426
# VEHICULAR_TRAILER                0.370  0.672  0.203  0.426  0.287   0.465
# WHEELCHAIR                       0.257  0.203  0.281  1.056  0.195   0.458
# WHEELED_DEVICE                   0.471  0.326  0.157  0.474  0.397   0.578
# WHEELED_RIDER                    0.245  0.329  0.293  0.458  0.196   0.397
# AVERAGE_METRICS                  0.486  0.368  0.267  0.510  0.395   0.599