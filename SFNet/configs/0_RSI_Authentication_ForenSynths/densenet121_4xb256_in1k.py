_base_ = [
    '../_base_/models/densenet/densenet121.py',
    # '../_base_/datasets/imagenet_bs64.py',
    # '../_base_/schedules/imagenet_bs256.py',
    # '../_base_/default_runtime.py',
    'my_base.py',
]

model = dict(
    type='ImageClassifier',
    backbone=dict(type='DenseNet', arch='121'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=2,
        in_channels=1024,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ))

# dataset settings
# train_dataloader = dict(batch_size=64)

# # schedule settings
# train_cfg = dict(by_epoch=True, max_epochs=90)

# NOTE: `auto_scale_lr` is for automatically scaling LR
# based on the actual training batch size.
# # base_batch_size = (4 GPUs) x (256 samples per GPU)
# auto_scale_lr = dict(base_batch_size=1024)
