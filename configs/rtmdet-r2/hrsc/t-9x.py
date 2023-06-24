_base_ = './rtmdet-r-l-9x.py'

checkpoint = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-tiny_imagenet_600e.pth'  # noqa

model = dict(
    backbone=dict(
        type='mmdet.CSPNeXt2',
        CSPBlock_kernel_size=5,
        deepen_factor=0.167,
        widen_factor=0.375,
        out_indices=(1, 2, 3, 4),
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone.',
            checkpoint=checkpoint
        )
    ),

    neck=dict(
        type='mmdet.CSPNeXtEPPAFPN',
        CSPBlock_kernel_size=5,
        in_channels=[48, 96, 192, 384],
        out_channels=96,
        num_csp_blocks=1
    ),

    bbox_head=dict(
        type='RotatedRTMDetTIDSepBNHead',
        in_channels=96,
        feat_channels=96,
        exp_on_reg=False,
        share_conv=True,
        use_ts_reg=True,
    ),

    train_cfg=dict(
        assigner=dict(
            type='mmdet.DynamicSoftLabelAssigner2',
            iou_calculator=dict(type='RBboxProbIoU2D'),
            topk=13,
            use_iou_cut=True,
            iou_cut_threshold=0.075
        ),
        allowed_border=-1,
        pos_weight=-1,
        debug=False
    ),
)

work_dir = './work_dirs/rtmdet-r2/hrsc/tiny-9x'

batch_size = 8
train_dataloader = dict(batch_size=batch_size, num_workers=batch_size)
val_dataloader = dict(batch_size=1, num_workers=2)

# training schedule, hrsc dataset is repeated 3 times, in
# `./_base_/hrsc_rr.py`, so the actual epoch = 3 * 3 * 12 = 9 * 12
max_epochs = 3 * 12

# hrsc dataset use larger learning rate for better performance
base_lr = (0.004 / 2) * 0.99

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        # use cosine lr from 54 to 108 epoch
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]

# optimizer
optim_wrapper = dict(optimizer=dict(lr=base_lr, weight_decay=(0.05)))