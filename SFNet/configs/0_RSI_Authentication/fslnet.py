_base_ = [
    'my_base.py',
]


model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='FSLNet',
        model="L"),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=2,
        in_channels=256,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1),
    ))