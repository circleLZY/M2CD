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
    type='DIEncoderDecoder_MoE',  # 使用支持 MoE 的双时相编码器-解码器结构
    data_preprocessor=data_preprocessor,
    pretrained=None,  # 预训练权重路径，如果有预训练权重可以填写路径
    backbone=dict(
        type='ConvNeXt_MoE',  # 使用 ConvNeXt 作为 Backbone
        in_channels=3,  # 输入通道数
        depths=[3, 3, 27, 3],  # 每个阶段的块数
        dims=[96, 192, 384, 768],  # 每个阶段的特征维度
        drop_path_rate=0.2,  # 随机深度率
        layer_scale_init_value=1e-6,  # Layer Scale 初始化值
        out_indices=[0, 1, 2, 3],  # 输出特征的阶段索引
        MoE_Block_inds=[[], [0, 2], [i*2 for i in range(14)], [0, 2]],  # 指定哪些 Block 使用 MoE
        num_experts=4,  # 专家数量
        top_k=2,  # 每个 token 选择的前 k 个专家
    ),
    # neck=dict(
    #     type='FeatureFusionNeck',  # 特征融合 Neck
    #     policy='concat'  # 特征融合策略为拼接
    # ),
    decode_head=dict(
        type='MLPSegHead',
        out_size=(128, 128),
        in_channels=[96 * 2, 192 * 2, 384 * 2, 768 * 2],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='mmseg.CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # 模型训练和测试配置
    train_cfg=dict(),
    test_cfg=dict(mode='whole')  # 测试模式为整个图像
)