# dataset settings
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

# data augmentation pipeline
input_size = 256
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="RandomResizedCrop", scale=input_size, crop_ratio_range=(0.8, 1.0)),
    dict(type="RandomFlip", prob=[0.5, 0.5], direction=["horizontal", "vertical"]),
    # 额外数据增强
    dict(type="ColorJitter", brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
    dict(
        type="RandomGrayscale",
        prob=0.2,
        keep_channels=True,
        channel_weights=(1.0, 1.0, 1.0),
    ),
    dict(
        type="GaussianBlur",
        prob=0.2,
        magnitude_range=(0.1, 2.0),
        magnitude_std="inf"),
    dict(type="PackInputs"),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="ResizeEdge", scale=input_size, edge="short"),
    dict(type="CenterCrop", crop_size=input_size),
    dict(type="PackInputs"),
]

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    dataset=dict(
        type=dataset_type,
        data_prefix=r"D:\Classification\ISPRS_RSI_Authentication\train",
        with_label=True,
        pipeline=train_pipeline,
    ),
    sampler=dict(type="DefaultSampler", shuffle=True),
    persistent_workers=True,
)

val_dataloader = dict(
    batch_size=batch_size*2,
    num_workers=num_workers,
    dataset=dict(
        type=dataset_type,
        data_prefix=r"D:\Classification\ISPRS_RSI_Authentication\val",
        with_label=True,
        pipeline=test_pipeline,
    ),
    sampler=dict(type="DefaultSampler", shuffle=False),
    persistent_workers=True,
)
val_evaluator = dict(type="Accuracy", topk=(1))

# If you want standard test, please manually configure the test dataset
test_dataloader = dict(
    batch_size=batch_size*2,
    num_workers=num_workers,
    dataset=dict(
        type=dataset_type,
        data_prefix=r"D:\Classification\ISPRS_RSI_Authentication\val",
        with_label=True,
        pipeline=test_pipeline,
    ),
    sampler=dict(type="DefaultSampler", shuffle=False),
)

test_evaluator = [
    dict(type='Accuracy', topk=(1)),  # Over Accuracy
    dict(type='SingleLabelMetric', items=['precision', 'recall', 'f1-score']),
]
