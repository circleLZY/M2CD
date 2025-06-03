_base_ = './changer_ex_s50_256x256_200k-cau.py'

model = dict(backbone=dict(depth=101, stem_channels=128))
