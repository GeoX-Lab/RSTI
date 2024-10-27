_base_ = [
    '../_base_/models/inception_v3.py',
    'my_base.py',
]


model = dict(
    backbone=dict(type='InceptionV3', num_classes=2, aux_logits=False),
    head=dict(
        type='ClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1)),
)
