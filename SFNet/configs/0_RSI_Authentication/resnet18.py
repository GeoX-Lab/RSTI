_base_ = [
    '../_base_/models/resnet18.py',
    'my_base.py',
]


# model = dict(head=dict(num_classes=100))
model = dict(
    head=dict(
        type='LinearClsHead',
        num_classes=2,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        # loss=dict(type='CrossEntropyLoss', class_weight=[0.6, 0.4]),
        # loss=dict(
        #     type='LabelSmoothLoss',
        #     label_smooth_val=0.1,
        #     mode='original',
        #     reduction='mean',
        #     class_weight=[0.7, 0.3]),
        topk=(1)
    ))
