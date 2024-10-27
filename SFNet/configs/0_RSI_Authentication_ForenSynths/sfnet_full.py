_base_ = [
    '../_base_/models/resnet50.py',
    'my_base.py',
]


model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='SFNet_full',
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

# model = dict(
#     head=dict(
#         type='LinearClsHead',
#         num_classes=2,
#         loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
#         topk=(1)
#     )
# )
