# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)

data_preprocessor = dict(
    type='DualInputSegDataPreProcessor',
    mean=[123.675, 116.28, 103.53] * 2,
    std=[58.395, 57.12, 57.375] * 2,
    bgr_to_rgb=True,
    size_divisor=32,
    pad_val=0,
    seg_pad_val=255,
    test_cfg=dict(size_divisor=32))

model = dict(
    type='DIEncoderDecoder_MoE_O2SP',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='IA_MobileNetV3Small_MoE_O2SP',
        arch='small',
        out_indices=(1, 5, 12),  # 选择 layer1, layer5, layer12 输出
        frozen_stages=-1,
        norm_cfg=norm_cfg,
        norm_eval=False,
        interaction_cfg=(None, None, None)
    ),
    
    decode_head=dict(
        type='MLPSegHead',
        out_size=(128, 128),
        in_channels=[16 * 2, 40 * 2, 576 * 2],  # 对应所选层的通道数*2
        in_index=[0, 1, 2],
        channels=256,
        dropout_ratio=0,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='mmseg.CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))