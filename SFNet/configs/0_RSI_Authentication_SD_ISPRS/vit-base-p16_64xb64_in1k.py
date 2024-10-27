_base_ = [
    '../_base_/models/vit-base-p16.py',
    'my_base.py',
    # '../_base_/datasets/imagenet_bs64_pil_resize_autoaug.py',
    # '../_base_/schedules/imagenet_bs4096_AdamW.py',
    # '../_base_/default_runtime.py'
]

# model setting
model = dict(
    head=dict(hidden_dim=3072, num_classes=2),
)

optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=0.003, weight_decay=0.3),
    # specific to vit pretrain
    paramwise_cfg=dict(custom_keys={
        '.cls_token': dict(decay_mult=0.0),
        '.pos_embed': dict(decay_mult=0.0)
    }),
    clip_grad=dict(max_norm=1.0),
)

# schedule setting
optim_wrapper = dict()

auto_scale_lr = dict(base_batch_size=4096)