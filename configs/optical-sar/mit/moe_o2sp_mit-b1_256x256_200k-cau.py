_base_ = ['./moe_o2sp_mit-b0_256x256_200k-cau.py']

checkpoint = '/nas/datasets/lzy/RS-SAR/checkpoints_cau/MiT-b1-MoE-MLPSeg/initial/best_mIoU_iter_185000.pth'  # noqa

model = dict(
    init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
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