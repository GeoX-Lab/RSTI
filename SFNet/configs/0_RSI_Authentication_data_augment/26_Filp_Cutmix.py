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
    dict(type="ResizeEdge", scale=input_size, edge="short"),
    dict(type="CenterCrop", crop_size=input_size),

    dict(type="RandomFlip", prob=[0.5, 0.5], direction=["horizontal", "vertical"]),

    dict(type="PackInputs"),
]

train_dataloader = dict(
    dataset=dict(pipeline=train_pipeline),
)

model = dict(
    train_cfg=dict(augments=[
        dict(type='CutMix', alpha=1.0),
        # dict(type='Mixup', alpha=0.8),
    ]),
)
