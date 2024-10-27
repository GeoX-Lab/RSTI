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

input_size = 256
train_pipeline = [
    dict(type="LoadImageFromFile"),

    dict(type="RandomResizedCrop", scale=input_size, crop_ratio_range=(0.08, 1.0)),  # Default settings
    dict(type="ColorJitter", brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),

    dict(type="PackInputs"),
]

train_dataloader = dict(
    dataset=dict(pipeline=train_pipeline),
)
