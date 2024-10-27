_base_ = [
    '../_base_/models/conformer/base-p16.py',
    # '../_base_/datasets/imagenet_bs64_swin_224.py',
    # '../_base_/schedules/imagenet_bs1024_adamw_conformer.py',
    # '../_base_/default_runtime.py',
    'my_base.py',
]

model = dict(
    head=dict(num_classes=2),
    train_cfg=dict(
        _delete_=True
    ),
)

batch_size = 16
num_workers = 4
dataset_type = "CustomDataset"
data_preprocessor = dict(
    num_classes=2,
    # RGB format normalization parameters
    mean=[128, 128, 128],
    std=[64, 64, 64],
    # convert image from BGR to RGB
    to_rgb=True,
)

bgr_mean = data_preprocessor['mean'][::-1]
bgr_std = data_preprocessor['std'][::-1]
input_size = 256
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        scale=input_size,
        crop_ratio_range=(0.8, 1.0),
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='RandAugment',
        policies='timm_increasing',
        num_policies=2,
        total_level=10,
        magnitude_level=9,
        magnitude_std=0.5,
        hparams=dict(
            pad_val=[round(x) for x in bgr_mean], interpolation='bicubic')),
    # dict(
    #     type='RandomErasing',
    #     erase_prob=0.25,
    #     mode='rand',
    #     min_area_ratio=0.02,
    #     max_area_ratio=1 / 3,
    #     fill_color=bgr_mean,
    #     fill_std=bgr_std),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ResizeEdge',
        scale=input_size,
        edge='short',
        backend='pillow',
        interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=input_size),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    dataset=dict(
        type=dataset_type,
        data_prefix=r"D:\Classification\ISPRS_RSI_Authentication\train",
        with_label=True,
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=batch_size*2,
    num_workers=num_workers,
    dataset=dict(
        type=dataset_type,
        data_prefix=r"D:\Classification\ISPRS_RSI_Authentication\val",
        with_label=True,
        pipeline=test_pipeline
    ),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
val_evaluator = dict(type='Accuracy', topk=(1))

# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
test_evaluator = [
    dict(type='Accuracy', topk=(1)),  # Over Accuracy
    dict(type='SingleLabelMetric', items=['precision', 'recall', 'f1-score']),
]

optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        # for batch in each gpu is 128, 8 gpu
        # lr = 5e-4 * 128 * 8 / 512 = 0.001
        lr=5e-4 * batch_size * 1 / 512,
        weight_decay=0.05,
        eps=1e-8,
        betas=(0.9, 0.999)),
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        custom_keys={
            '.cls_token': dict(decay_mult=0.0),
        }),
)
auto_scale_lr = dict(base_batch_size=1024)