_base_ = [
    '../_base_/datasets/0_RSI_Authentication.py',
    '../_base_/models/resnet50.py',
    '../_base_/default_runtime.py'
]


# model
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


# optimizer
lr = 1e-4
optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0))
# optim_wrapper = dict(
#     optimizer=dict(type='SGD', lr=lr, momentum=0.9, weight_decay=0.0001))
max_epochs = 200
warmup_epochs = 0
param_scheduler = [
    # warm up learning rate scheduler
    # dict(
    #     type='LinearLR',
    #     start_factor=0.0001,
    #     by_epoch=True,
    #     begin=0,
    #     end=warmup_epochs,
    #     # update by iter
    #     convert_to_iter_based=True),
    # main learning rate scheduler
    dict(
        type='CosineAnnealingLR',
        T_max=max_epochs-warmup_epochs,
        eta_min=1.0e-6,
        by_epoch=True,
        begin=warmup_epochs,
        end=max_epochs)
]

train_cfg = dict(by_epoch=True, max_epochs=max_epochs, val_interval=5)
val_cfg = dict()
test_cfg = dict()
auto_scale_lr = dict(base_batch_size=128)

# configure default hooks
default_hooks = dict(
    # save checkpoint
    checkpoint=dict(
        type='CheckpointHook',
        save_best='auto',
        interval=1,
        max_keep_ckpts=1,
        rule='greater'),
)
