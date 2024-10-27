_base_ = [
    '../_base_/models/convnext/convnext-base.py',
    'my_base.py',
    # '../_base_/datasets/imagenet_bs64_swin_224.py',
    # '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    # '../_base_/default_runtime.py',
]
#
model = dict(
    type='ImageClassifier',
    backbone=dict(type='ConvNeXt', arch='base', drop_path_rate=0.5),
    head=dict(
        type='LinearClsHead',
        num_classes=2,
        in_channels=1024,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
        init_cfg=None,
    ),
    init_cfg=dict(
        type='TruncNormal', layer=['Conv2d', 'Linear'], std=.02, bias=0.),
    # train_cfg=dict(augments=[
    #     dict(type='Mixup', alpha=0.8),
    #     dict(type='CutMix', alpha=1.0),
    # ]),
)
