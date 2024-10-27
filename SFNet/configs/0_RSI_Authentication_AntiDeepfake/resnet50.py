_base_ = [
    '../_base_/models/resnet50.py',
    'my_base.py',
]


model = dict(
    # backbone=dict(
    #     frozen_stages=3,
    #     # init_cfg=dict(type='Pretrained', checkpoint=r'N:\Model\RSI_Authentication\mmpretrain\resnet50_8xb32_in1k_20210831-ea4938fc.pth', prefix='backbone.')
    #     init_cfg=dict(type='Pretrained', checkpoint='', prefix='backbone.')
    # ),
    head=dict(
        type='LinearClsHead',
        num_classes=2,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1)
    )
)
