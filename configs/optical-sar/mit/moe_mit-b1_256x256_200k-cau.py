_base_ = ['./moe_mit-b0_256x256_200k-cau.py']

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b1_20220624-02e5a6a1.pth'  # noqa

# model settings
model = dict(
    pretrained=checkpoint,
    backbone=dict(
        embed_dims=64, num_heads=[1, 2, 5, 8], num_layers=[2, 2, 2, 2],
        interaction_cfg=(
            dict(type='MoE_layer', moe_cfg=dict(num_experts=4, top_k=2, in_channels=64, mid_channels=64*4, pw_conv='nn.Linear', act_cfg=dict(type='GELU'), noisy_gating=True, use_grn=False, gating='cosine')),
            dict(type='MoE_layer', moe_cfg=dict(num_experts=4, top_k=2, in_channels=128, mid_channels=128*4, pw_conv='nn.Linear', act_cfg=dict(type='GELU'), noisy_gating=True, use_grn=False, gating='cosine')),
            dict(type='MoE_layer', moe_cfg=dict(num_experts=4, top_k=2, in_channels=320, mid_channels=320*4, pw_conv='nn.Linear', act_cfg=dict(type='GELU'), noisy_gating=True, use_grn=False, gating='cosine')),
            dict(type='MoE_layer', moe_cfg=dict(num_experts=4, top_k=2, in_channels=512, mid_channels=512*4, pw_conv='nn.Linear', act_cfg=dict(type='GELU'), noisy_gating=True, use_grn=False, gating='cosine'))
        ),
    ),
    decode_head=dict(
        in_channels=[64 * 2, 128 * 2, 320 * 2, 512 * 2],)
)