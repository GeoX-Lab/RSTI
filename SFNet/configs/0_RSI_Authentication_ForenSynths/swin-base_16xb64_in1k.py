_base_ = [
    '../_base_/models/swin_transformer/base_224.py',
    'my_base.py',
]

# # schedule settings
# optim_wrapper = dict(clip_grad=dict(max_norm=5.0))
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='SwinTransformer', arch='base', img_size=224, drop_path_rate=0.5),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=2,
        in_channels=1024,
        init_cfg=None,  # suppress the default init_cfg of LinearClsHead.
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
        cal_acc=False),
    init_cfg=[
        dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
        dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
    ],
    # train_cfg=dict(augments=[
    #     dict(type='Mixup', alpha=0.8),
    #     dict(type='CutMix', alpha=1.0)
    # ]),
)
