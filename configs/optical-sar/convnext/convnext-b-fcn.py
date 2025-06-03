_base_ = [
    '/home/lzy/proj/rmcd-kd/configs/_base_/models/convnext/convnext-b-fcn.py',
    '/home/lzy/proj/rmcd-kd/configs/common/standard_256x256_200k_cau.py']

# crop_size = (512, 512)


# model = dict(
#     # pretrained=checkpoint,
#     backbone=dict(
#         interaction_cfg=(
#             None,
#             dict(type='SpatialExchange', p=1/2),
#             dict(type='ChannelExchange', p=1/2),
#             dict(type='ChannelExchange', p=1/2))
#     ),
#     init_cfg=dict(type='Pretrained', prefix='backbone', checkpoint='/root/siton-gpfs-archive/yuxuanli/mmpretrain/work_dirs/convnext_t_sar/epoch_100.pth'
# )