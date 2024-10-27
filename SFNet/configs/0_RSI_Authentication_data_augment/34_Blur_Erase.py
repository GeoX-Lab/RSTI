_base_ = './sfnet_full.py'

# data augmentation pipeline

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
    dict(type="LoadImageFromFile"),
    dict(type="ResizeEdge", scale=input_size, edge="short"),
    dict(type="CenterCrop", crop_size=input_size),

    dict(
        type="GaussianBlur",
        prob=0.2,
        magnitude_range=(0.1, 2.0),
        magnitude_std="inf"),
    dict(
        type='RandomErasing',
        erase_prob=0.25,
        mode='rand',
        min_area_ratio=0.02,
        max_area_ratio=1 / 3,
        fill_color=bgr_mean,
        fill_std=bgr_std),

    dict(type="PackInputs"),
]

train_dataloader = dict(
    dataset=dict(pipeline=train_pipeline),
)
