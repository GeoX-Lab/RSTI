_base_ = [
    '../_base_/models/resnet/resnet18_freq.py',
    # '../_base_/models/resnet18_cifar.py',
    'my_base.py',
]


# model = dict(head=dict(num_classes=100))
model = dict(
    head=dict(
        type='LinearClsHead',
        num_classes=2,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1)
    ))
