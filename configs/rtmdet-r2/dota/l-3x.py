_base_ = './rtmdet-r-l-3x.py'

checkpoint = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-l_8xb256-rsb-a1-600e_in1k-6a760974.pth'  # noqa

model = dict(
    backbone=dict(
        type='mmdet.CSPNeXt2',
        CSPBlock_kernel_size=5,
        deepen_factor=1,
        widen_factor=1,
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
        in_channels=[128, 256, 512, 1024],
        out_channels=256,
        num_csp_blocks=3
    ),

    bbox_head=dict(
        type='RotatedRTMDetTIDSepBNHead',
        in_channels=256,
        feat_channels=256,
        exp_on_reg=False,
        share_conv=True,
        use_ts_reg=True,
    ),

    train_cfg=dict(
        assigner=dict(
            type='mmdet.DynamicSoftLabelAssigner2',
            iou_calculator=dict(type='RBboxProbIoU2D'),
            use_iou_cut=True,
            iou_cut_threshold=0.075
        ),
    ),
)

# ------------------------------------------------------------------------------------
work_dir = './work_dirs/rtmdet-r2/dota/large-3x/'

gpu_num = 2
batch_size = 4

# ------------------------------------------------------------------------------------
train_dataloader = dict(batch_size=batch_size, num_workers=batch_size)
val_dataloader = dict(batch_size=1, num_workers=1)

max_epochs = 3 * 12
base_lr = 0.004 / 16

base_lr = (float(batch_size * gpu_num)/8.0) * base_lr

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]

optim_wrapper = dict(
    optimizer=dict(
        type='AdamW', 
        lr=base_lr, 
        weight_decay=0.05*2.375
    )
)

test_evaluator = dict(outfile_prefix=work_dir + 'e36-Task1')