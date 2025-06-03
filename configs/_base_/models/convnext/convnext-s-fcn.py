# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='DualInputSegDataPreProcessor',
    mean=[123.675, 116.28, 103.53] * 2,  # 双时相输入，均值和标准差乘以 2
    std=[58.395, 57.12, 57.375] * 2,
    bgr_to_rgb=True,
    size_divisor=32,
    pad_val=0,
    seg_pad_val=255,
    test_cfg=dict(size_divisor=32))

# 模型配置
model = dict(
    type='DIEncoderDecoder',  # 使用双时相编码器-解码器结构
    data_preprocessor=data_preprocessor,
    pretrained=None,  # 预训练权重路径，如果有预训练权重可以填写路径
    backbone=dict(
        type='ConvNeXt',  # 使用 ConvNeXt 作为 Backbone
        in_channels=3,  # 输入通道数
        depths=[3, 3, 27, 3], 
        dims=[96, 192, 384, 768], 
        drop_path_rate=0.4,
        layer_scale_init_value=1e-6,
        out_indices=[0, 1, 2, 3]
    ),
    # neck=dict(
    #     type='FeatureFusionNeck',  # 特征融合 Neck
    #     policy='concat'  # 特征融合策略为拼接
    # ),
    decode_head=dict(
        type='mmseg.FCNHead',  # 使用 FCNHead 作为解码头
        in_channels=768 * 2,  # 输入通道数（ConvNeXt 最后一层特征维度为 768，双时相拼接后为 768 * 2）
        channels=512,  # 中间通道数
        num_convs=2,  # 卷积层数
        concat_input=False,  # 是否拼接输入特征
        num_classes=2,  # 输出类别数（背景、无损房屋、轻度损伤房屋、重度损伤房屋）
        norm_cfg=norm_cfg,  # 归一化配置
        align_corners=False,  # 是否对齐角点
        loss_decode=dict(
            type='mmseg.CrossEntropyLoss',  # 交叉熵损失
            use_sigmoid=False,  # 是否使用 sigmoid
            loss_weight=1.0  # 损失权重
        )
    ),
    # 模型训练和测试配置
    train_cfg=dict(),
    test_cfg=dict(mode='whole')  # 测试模式为整个图像
)