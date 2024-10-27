_base_ = [
    'my_base.py',
]


model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='FreqNet',
        depth=50,
        num_stages=4,
        out_indices=(1, ),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=2,
        in_channels=512,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1),
    ))
