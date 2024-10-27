_base_ = [
    '../_base_/models/vgg16.py',
    'my_base.py',
]

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(type='VGG', depth=16, num_classes=2),
    neck=None,
    head=dict(
        type='ClsHead',
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1),
    ))
