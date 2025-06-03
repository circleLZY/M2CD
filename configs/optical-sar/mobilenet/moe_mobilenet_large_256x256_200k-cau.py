_base_ = [
    '/home/lzy/proj/rmcd-kd/configs/_base_/models/mobilenet/moe_mobilenet_large.py',
    '/home/lzy/proj/rmcd-kd/configs/common/standard_256x256_200k_cau.py']

crop_size = (256, 256)

checkpoint = 'https://download.openmmlab.com/mmclassification/v0/mobilenet_v3/mobilenet-v3-large_8xb128_in1k_20221114-0ed9ed9a.pth'  # noqa

model = dict(
    pretrained=checkpoint,
    backbone=dict(
        interaction_cfg=(
            dict(  # 对应 layer1 的输出
                type='MoE_layer_3', 
                moe_cfg=dict(
                    num_experts=4, 
                    top_k=2, 
                    in_channels=24, 
                    mid_channels=96, 
                    pw_conv='nn.Linear', 
                    act_cfg=dict(type='GELU'), 
                    noisy_gating=True, 
                    use_grn=False, 
                    gating='cosine'
                )
            ),
            dict(  # 对应 layer5 的输出
                type='MoE_layer_3', 
                moe_cfg=dict(
                    num_experts=4, 
                    top_k=2, 
                    in_channels=40, 
                    mid_channels=160, 
                    pw_conv='nn.Linear', 
                    act_cfg=dict(type='GELU'), 
                    noisy_gating=True, 
                    use_grn=False, 
                    gating='cosine'
                )
            ),
            dict(  # 对应 layer12 的输出
                type='MoE_layer_3', 
                moe_cfg=dict(
                    num_experts=4, 
                    top_k=2, 
                    in_channels=160, 
                    mid_channels=640, 
                    pw_conv='nn.Linear', 
                    act_cfg=dict(type='GELU'), 
                    noisy_gating=True, 
                    use_grn=False, 
                    gating='cosine'
                )
            ),
        ),
    ),
    decode_head=dict(
        num_classes=2,
        sampler=dict(type='mmseg.OHEMPixelSampler', thresh=0.7, min_kept=100000)),
)

train_pipeline = [
    dict(type='MultiImgLoadImageFromFile'),
    dict(type='MultiImgLoadAnnotations'),
    dict(type='MultiImgRandomRotFlip', rotate_prob=0.5, flip_prob=0.5, degree=(-20, 20)),
    dict(type='MultiImgRandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='MultiImgExchangeTime', prob=0.5),
    dict(
        type='MultiImgPhotoMetricDistortion',
        brightness_delta=10,
        contrast_range=(0.8, 1.2),
        saturation_range=(0.8, 1.2),
        hue_delta=10),
    dict(type='MultiImgPackSegInputs')
]

train_dataloader = dict(
    dataset=dict(pipeline=train_pipeline))

# optimizer
optimizer=dict(
    type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.01)

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=optimizer,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))