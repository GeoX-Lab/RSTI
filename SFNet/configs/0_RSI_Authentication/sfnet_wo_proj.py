_base_ = [
    # '../_base_/datasets/0_RSI_Authentication.py',
    '../_base_/models/resnet50.py',
    # '../_base_/default_runtime.py'
    'my_base.py',
]


# model
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='SFNet_wo_proj',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=2,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1)
    ))
