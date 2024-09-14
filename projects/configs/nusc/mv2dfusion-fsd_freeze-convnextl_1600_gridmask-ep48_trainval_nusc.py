_base_ = [
    '../_base_/datasets/nus-3d.py',
    '../_base_/default_runtime.py',
]
plugin = True
plugin_dir = [
    'projects/mmdet3d_plugin/',
    'projects/fsdv2/'
]

class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

# FSDv2 setting
voxel_size = [0.2, 0.2, 8]
point_cloud_range = [-54.4, -54.4, -5.0, 54.4, 54.4, 3.0]
sparse_shape = [40, 544, 544]
target_sparse_shape = [20, 272, 272]
fsd_class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]
seg_voxel_size = (0.2, 0.2, 0.2)
virtual_voxel_size=(0.4, 0.4, 0.4) #(1024, 1024, 16)
group1 = ['car']
group2 = ['truck', 'construction_vehicle']
group3 = ['bus', 'trailer']
group4 = ['barrier']
group5 = ['motorcycle', 'bicycle']
group6 = ['pedestrian', 'traffic_cone']
group_names = [group1, group2, group3, group4, group5, group6]
seg_score_thresh = [0.2, ] * 3 + [0.1, ] * 3
group_lens = [len(group1), len(group2), len(group3), len(group4), len(group5), len(group6)]
head_group1 = fsd_class_names[:5]
head_group2 = fsd_class_names[5:]
tasks = [
    dict(class_names=head_group1),
    dict(class_names=head_group2),
]

queue_length = 1
num_frame_losses = 1

roi_size = 7
roi_strides = [4, 8, 16, 32, 64]
model = dict(
    type='MV2DFusion',
    dataset='nuscenes',
    num_frame_head_grads=num_frame_losses,
    num_frame_backbone_grads=num_frame_losses,
    num_frame_losses=num_frame_losses,
    position_level=2,
    use_grid_mask=True,

    loss_weight_3d=0.1,
    loss_weight_pts=1.,
    gt_mono_loss=True,

    img_backbone=dict(
        type='ConvNeXt',
        arch='large',
        out_indices=[0, 1, 2, 3],
        drop_path_rate=0.4,
        layer_scale_init_value=0.,
        gap_before_final_norm=False,
        use_grn=True,
        with_cp=True,
    ),
    img_neck=dict(
        type='FPN',
        in_channels=[192, 384, 768, 1536],
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
            type='CascadeRoIHead',
            num_stages=3,
            stage_loss_weights=[1, 0.5, 0.25],
            bbox_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
                out_channels=256,
                featmap_strides=roi_strides[:-1]),
            bbox_head=[
                dict(
                    type='Shared2FCBBoxHead',
                    in_channels=256,
                    fc_out_channels=1024,
                    roi_feat_size=7,
                    num_classes=10,
                    bbox_coder=dict(
                        type='DeltaXYWHBBoxCoder',
                        target_means=[0., 0., 0., 0.],
                        target_stds=[0.1, 0.1, 0.2, 0.2]),
                    reg_class_agnostic=True,
                    loss_cls=dict(
                        type='CrossEntropyLoss',
                        use_sigmoid=False,
                        loss_weight=1.0),
                    # loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                    #                loss_weight=1.0),
                    reg_decoded_bbox=True,
                    loss_bbox=dict(type='GIoULoss', loss_weight=10.0)
                ),
                dict(
                    type='Shared2FCBBoxHead',
                    in_channels=256,
                    fc_out_channels=1024,
                    roi_feat_size=7,
                    num_classes=10,
                    bbox_coder=dict(
                        type='DeltaXYWHBBoxCoder',
                        target_means=[0., 0., 0., 0.],
                        target_stds=[0.05, 0.05, 0.1, 0.1]),
                    reg_class_agnostic=True,
                    loss_cls=dict(
                        type='CrossEntropyLoss',
                        use_sigmoid=False,
                        loss_weight=1.0),
                    reg_decoded_bbox=True,
                    loss_bbox=dict(type='GIoULoss', loss_weight=10.0)
                ),
                dict(
                    type='Shared2FCBBoxHead',
                    in_channels=256,
                    fc_out_channels=1024,
                    roi_feat_size=7,
                    num_classes=10,
                    bbox_coder=dict(
                        type='DeltaXYWHBBoxCoder',
                        target_means=[0., 0., 0., 0.],
                        target_stds=[0.033, 0.033, 0.067, 0.067]),
                    reg_class_agnostic=True,
                    loss_cls=dict(
                        type='CrossEntropyLoss',
                        use_sigmoid=False,
                        loss_weight=1.0),
                    # loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                    #                loss_weight=1.0),
                    reg_decoded_bbox=True,
                    loss_bbox=dict(type='GIoULoss', loss_weight=10.0)
                )
            ],
            mask_head=None,
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
                allowed_border=0,
                pos_weight=-1,
                debug=False),
            rpn_proposal=dict(
                nms_pre=2000,
                nms_post=2000,
                max_per_img=2000,
                nms=dict(type='nms', iou_threshold=0.7),
                min_bbox_size=0),
            rcnn=[
                dict(
                    assigner=dict(
                        type='MaxIoUAssigner',
                        pos_iou_thr=0.5,
                        neg_iou_thr=0.5,
                        min_pos_iou=0.5,
                        match_low_quality=False,
                        ignore_iof_thr=-1),
                    sampler=dict(
                        type='RandomSampler',
                        num=512,
                        pos_fraction=0.25,
                        neg_pos_ub=-1,
                        add_gt_as_proposals=True),
                    mask_size=28,
                    pos_weight=-1,
                    debug=False),
                dict(
                    assigner=dict(
                        type='MaxIoUAssigner',
                        pos_iou_thr=0.6,
                        neg_iou_thr=0.6,
                        min_pos_iou=0.6,
                        match_low_quality=False,
                        ignore_iof_thr=-1),
                    sampler=dict(
                        type='RandomSampler',
                        num=512,
                        pos_fraction=0.25,
                        neg_pos_ub=-1,
                        add_gt_as_proposals=True),
                    mask_size=28,
                    pos_weight=-1,
                    debug=False),
                dict(
                    assigner=dict(
                        type='MaxIoUAssigner',
                        pos_iou_thr=0.7,
                        neg_iou_thr=0.7,
                        min_pos_iou=0.7,
                        match_low_quality=False,
                        ignore_iof_thr=-1),
                    sampler=dict(
                        type='RandomSampler',
                        num=512,
                        pos_fraction=0.25,
                        neg_pos_ub=-1,
                        add_gt_as_proposals=True),
                    mask_size=28,
                    pos_weight=-1,
                    debug=False)
            ]),
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
        prob_bin=50,
        depth_range=[0.1, 90],
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
    # fsdv2
    pts_backbone=dict(
        type='SingleStageFSDV2',
        freeze=True,
        norm_eval=True,
        segmentor=dict(
            type='VoteSegmentor',
            tanh_dims=[],
            voxel_layer=dict(
                voxel_size=seg_voxel_size,
                max_num_points=-1,
                point_cloud_range=point_cloud_range,
                max_voxels=(-1, -1)
            ),
            voxel_encoder=dict(
                type='DynamicScatterVFE',
                in_channels=5,
                feat_channels=[64, 64],
                voxel_size=seg_voxel_size,
                with_cluster_center=True,
                with_voxel_center=True,
                point_cloud_range=point_cloud_range,
                norm_cfg=dict(type='MyBN1d', eps=1e-3, momentum=0.01),
                unique_once=True,
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
                    loss_weight=10.0),
                loss_vote=dict(
                    type='L1Loss',
                    loss_weight=1.0),
            ),
            train_cfg=dict(
                point_loss=True,
                score_thresh=seg_score_thresh, # for training log
                class_names=fsd_class_names, # for training log
                group_names=group_names,
                group_lens=group_lens,
            ),
        ),
        virtual_point_projector=dict(
            in_channels=83 + 64,
            hidden_dims=[64, 64],
            norm_cfg=dict(type='MyBN1d'),

            ori_in_channels=67 + 64,
            ori_hidden_dims=[64, 64],

            # TODO: optional
            recover_in_channels=128 + 3, # with point2voxel offset
            recover_hidden_dims=[128, 128],
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
            bbox_coder=dict(type='BasePointBBoxCoder', code_size=10),
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=4.0),
            loss_center=dict(type='L1Loss', loss_weight=0.5),
            loss_size=dict(type='L1Loss', loss_weight=0.5),
            loss_rot=dict(type='L1Loss', loss_weight=0.2),
            loss_vel=dict(type='L1Loss', loss_weight=0.2),
            in_channel=128,
            shared_mlp_dims=[256, 256],
            train_cfg=None,
            test_cfg=None,
            norm_cfg=dict(type='MyBN1d'),
            tasks=tasks,
            class_names=fsd_class_names,
            common_attrs=dict(
                center=(3, 2, 128), dim=(3, 2, 128), rot=(2, 2, 128), vel=(2, 2, 128)
                # (out_dim, num_layers, hidden_dim)
            ),
            num_cls_layer=2,
            cls_hidden_dim=128,
            separate_head=dict(
                type='FSDSeparateHead',
                norm_cfg=dict(type='MyBN1d'),
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
            disable_pretrain_topks=[500, ] * len(class_names),
        ),
        test_cfg=dict(
            score_thresh=seg_score_thresh,
            batched_group_sample=True,
            offset_weight='max',
            class_names=fsd_class_names,
            group_names=[group1, group2, group3, group4, group5, group6],
            use_rotate_nms=True,
            nms_pre=-1,
            nms_thr=0.25,
            # score_thr=0.1,
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
        pts_use_cat=False,
    ),
    # fusion head
    fusion_bbox_head=dict(
        type='MV2DFusionHead',
        prob_bin=50,
        post_bev_nms_ops=[0],
        num_classes=10,
        in_channels=256,
        num_query=300,
        memory_len=12 * 256,
        topk_proposals=256,
        num_propagated=256,
        with_ego_pos=True,
        match_with_velo=True,
        scalar=10,  ##noise groups
        noise_scale=1.0,
        dn_weight=1.0,  ##dn loss weight
        split=0.75,  ###positive rate
        code_weights=[2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
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
                            num_cams=6,
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
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=10),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0), ),
    train_cfg=dict(fusion=dict(
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
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
    "resize_lim": (0.75, 1.35),
    "final_dim": (640, 1600),
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (0.0, 0.0),
    "H": 900,
    "W": 1600,
    "rand_flip": True,
}

dataset_type = 'CustomNuScenesDataset'
data_root = './data/nuscenes/'

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
        pad_empty_sweeps=True,
        remove_close=True,
        file_client_args=file_client_args),
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='ResizeCropFlipRotImage', data_aug_conf=ida_aug_conf, training=False),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='NormalizePoints'),
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
    samples_per_gpu=1,
    workers_per_gpu=4,
    val=dict(
        type=dataset_type,
        pipeline=test_pipeline,
        collect_keys=collect_keys + ['img', 'img_metas'],
        queue_length=queue_length,
        ann_file=data_root + 'nuscenes2d_temporal_infos_val.pkl',
        classes=class_names,
        modality=input_modality),
    test=dict(
        type=dataset_type,
        pipeline=test_pipeline,
        collect_keys=collect_keys + ['img', 'img_metas'],
        queue_length=queue_length,
        ann_file=data_root + 'nuscenes2d_temporal_infos_test.pkl',
        classes=class_names,
        modality=input_modality),
    shuffler_sampler=dict(type='InfiniteGroupEachSampleInBatchSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)

# TODO: this config is released for inference only
#   the training code has not been completely checked yet

# Accumulating metric data
# Calculating metrics
# Saving metrics to: /tmp/tmpsejislvg/nuscenes-metrics
# mAP: 0.7448
# mATE: 0.2446
# mASE: 0.2285
# mAOE: 0.2688
# mAVE: 0.1990
# mAAE: 0.1149
# NDS: 0.7668
# Eval time: 283.5s
# Completed evaluation for test phase