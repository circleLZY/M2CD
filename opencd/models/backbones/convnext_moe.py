# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from timm.models.layers import trunc_normal_, DropPath
import math
from mmengine.model import (constant_init, normal_init, trunc_normal_init)
from mmengine.model import BaseModule
from mmcv.cnn import build_activation_layer
from opencd.registry import MODELS


class FFN(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 pw_conv,
                 act_cfg=dict(type='GELU'),
                 use_grn=False):
        super().__init__()
        self.pointwise_conv1 = pw_conv(in_channels, mid_channels)
        self.act = build_activation_layer(act_cfg)
        self.pointwise_conv2 = pw_conv(mid_channels, in_channels)

        if use_grn:
            self.grn = GRN(mid_channels)
        else:
            self.grn = None
    def forward(self, x):
        x = self.pointwise_conv1(x)
        # print(x)
        x = self.act(x)
        if self.grn is not None:
            x = self.grn(x, data_format='channel_last')
         
        x = self.pointwise_conv2(x) 
        return x
    
    
class GRN(nn.Module):
    def __init__(self, in_channels, eps=1e-6):
        super().__init__()
        self.in_channels = in_channels
        self.gamma = nn.Parameter(torch.zeros(in_channels))
        self.beta = nn.Parameter(torch.zeros(in_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor, data_format='channel_first'):
        if data_format == 'channel_last':
            gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
            nx = gx / (gx.mean(dim=-1, keepdim=True) + self.eps)
            x = self.gamma * (x * nx) + self.beta + x
        elif data_format == 'channel_first':
            gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
            nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)
            x = self.gamma.view(1, -1, 1, 1) * (x * nx) + self.beta.view(
                1, -1, 1, 1) + x
        return x


class CosineTopKGate(nn.Module):
    def __init__(self, model_dim, num_global_experts, init_t=0.5):
        super(CosineTopKGate, self).__init__()
        proj_dim = min(model_dim // 2, 256)
        self.temperature = nn.Parameter(torch.log(torch.full([1], 1.0 / init_t)), requires_grad=True)
        self.cosine_projector = nn.Linear(model_dim, proj_dim)
        self.sim_matrix = nn.Parameter(torch.randn(size=(proj_dim, num_global_experts)), requires_grad=True)
        self.clamp_max = torch.log(torch.tensor(1. / 0.01)).item()
        nn.init.normal_(self.sim_matrix, 0, 0.01)

    def forward(self, x):
        # x 的形状是 (batch_size * height * width, model_dim)
        cosine_projector = self.cosine_projector
        sim_matrix = self.sim_matrix
        logits = torch.matmul(F.normalize(cosine_projector(x), dim=1),
                             F.normalize(sim_matrix, dim=0))
        logit_scale = torch.clamp(self.temperature, max=self.clamp_max).exp()
        logits = logits * logit_scale
        return logits
    
    
class MoE_layer(nn.Module):
    def __init__(self, moe_cfg):
        super(MoE_layer, self).__init__()
        self.noisy_gating = moe_cfg['noisy_gating']
        self.num_experts = moe_cfg['num_experts']
        self.input_size = moe_cfg['in_channels']
        self.k = moe_cfg['top_k']
        
        # 动态加载 pw_conv
        if isinstance(moe_cfg['pw_conv'], str):
            moe_cfg['pw_conv'] = eval(moe_cfg['pw_conv'])
        
        # 实例化专家
        self.gating = moe_cfg['gating']
        self.experts = nn.ModuleList([
            FFN(
                in_channels=self.input_size,
                mid_channels=moe_cfg['mid_channels'],
                pw_conv=moe_cfg['pw_conv'],
                act_cfg=moe_cfg['act_cfg'],
                use_grn=moe_cfg['use_grn']
            ) for i in range(self.num_experts)
        ])
        self.infer_expert = None
 
        if moe_cfg['gating'] == 'linear':
            self.w_gate = nn.Parameter(torch.zeros(self.input_size, self.num_experts), requires_grad=True)
        elif moe_cfg['gating'] == 'cosine':
            self.w_gate = CosineTopKGate(self.input_size, self.num_experts)
        self.w_noise = nn.Parameter(torch.zeros(self.input_size, self.num_experts), requires_grad=True)

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(-1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))

        assert(self.k <= self.num_experts)

    def cv_squared(self, x):
        eps = 1e-10
        # if only num_experts = 1
        if x.shape[0] == 1:
            return torch.Tensor([0])
        if len(x.shape) == 2:
            x = x.sum(dim=0)
        return x.float().var() / (x.float().mean()**2 + eps)

    def _gates_to_load(self, gates):
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten() # (bs x m)
        threshold_positions_if_in = torch.arange(batch) * m + self.k # bs
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in.to(top_values_flat.device)), 1)

        if len(noisy_values.shape) == 3:
            threshold_if_in = threshold_if_in.unsqueeze(1)

        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat,0 , threshold_positions_if_out.to(top_values_flat.device)), 1)
        if len(noisy_values.shape) == 3:
            threshold_if_out = threshold_if_out.unsqueeze(1)

        # is each value currently in the top k.
        normal = Normal(self.mean.to(noise_stddev.device), self.std.to(noise_stddev.device))
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def random_k_gating(self, features, train):
        if train:
            idx = torch.randint(0, self.num_experts, 1)
            results = self.experts[idx](features)

        else:
            results = []
            for i in range(self.num_experts):
                tmp = self.num_experts[i](features)
                results.append(tmp)
            
            results = torch.stack(results, dim=0).mean(dim=0)

        return results

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        if self.gating == 'linear':
            clean_logits = x @ self.w_gate
        elif self.gating == 'cosine':
            clean_logits = self.w_gate(x)

        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon) * train)
            noisy_logits = clean_logits + ( torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim= -1)  
        
        top_k_logits = top_logits[:, :self.k] if len(top_logits.shape) == 2 else top_logits[:, :, :self.k]    
        top_k_indices = top_indices[:, :self.k] if len(top_indices.shape) == 2 else top_indices[:, :, :self.k]
        
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
       
        gates = zeros.scatter(-1, top_k_indices, top_k_gates)  

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load


    def forward(self, x, loss_coef=1e-2):
        train = self.training 
        
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        x_shape = x.shape 
        x = x.reshape(-1,x.shape[-1])
        gates, load = self.noisy_top_k_gating(x, train)
        importance = gates.sum(dim=0)
        
        # calculate loss
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef

        dispatcher = SparseDispatcher(self.num_experts, gates)
        
        expert_inputs = dispatcher.dispatch(x) 
        gates = dispatcher.expert_to_gates() 
        expert_outputs = [self.experts[i](expert_inputs[i]).reshape(-1, x_shape[-1]) for i in range(self.num_experts)]
        y = dispatcher.combine(expert_outputs)
        y = y.reshape(x_shape)
        # assert False, (y.shape, y[0][0][0])
        return y, loss
    

class SparseDispatcher(object):
    def __init__(self, num_experts, gates):
        self._gates = gates
        self._num_experts = num_experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)  # torch.nonzero: 
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = sorted_experts[index_sorted_experts[:, 1],0]
        # calculate num samples that each expert gets
        self._part_sizes = list((gates > 0).sum(0).cpu().numpy())
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)


    def combine(self, expert_out, multiply_by_gates=True):
        stitched = torch.cat(expert_out, 0)

        if multiply_by_gates:
            if len(stitched.shape) == 3:
                stitched = stitched.mul(self._nonzero_gates.unsqueeze(1))
            else:
                stitched = stitched.mul(self._nonzero_gates)

        if len(stitched.shape) == 3:
            zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), expert_out[-1].size(-1), requires_grad=True, device=stitched.device)
        else:
            zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True, device=stitched.device)
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        return combined

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
          a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[expert_batch_size_i]`
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)
    

class ConvNeXtBlock(nn.Module):
    r""" ConvNeXt Block with optional MoE (Mixture of Experts) module.
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        MoE_cfg (dict, optional): Configuration for MoE module. Default: None.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, MoE_cfg=None):
        super().__init__()
        self.depthwise_conv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pointwise_conv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pointwise_conv2 = nn.Linear(4 * dim, dim)
        
        # MoE module
        self.MoE_cfg = MoE_cfg
        if MoE_cfg is not None:
            self.ffn = MoE_layer(MoE_cfg)
        else:
            self.ffn = nn.Sequential(
                self.pointwise_conv1,
                self.act,
                self.pointwise_conv2
            )
        
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.depthwise_conv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        
        # Apply MoE or standard FFN
        if self.MoE_cfg is not None:
            x, loss = self.ffn(x)
        else:
            x = self.ffn(x)
            loss = None
        
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        if self.gamma is not None:
            x = x.mul(self.gamma.view(1, -1, 1, 1))

        x = input + self.drop_path(x)
        return x, loss
    

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


@MODELS.register_module()
class ConvNeXt_MoE(BaseModule):
    def __init__(self, 
                 in_channels=3, 
                 depths=[3, 3, 9, 3], 
                 dims=[96, 192, 384, 768], 
                 drop_path_rate=0., 
                 layer_scale_init_value=1e-6, 
                 out_indices=[0, 1, 2, 3],
                 pretrained=None, 
                 init_cfg=None,
                 MoE_cfg=None,
                 MoE_Block_inds=None,  # 指定哪些 Block 使用 MoE
                 num_experts=8,  # 专家数量
                 top_k=2):  # 每个 token 选择的前 k 个专家
        super().__init__(init_cfg=init_cfg)

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')

        # Stem and downsampling layers
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        # Stages
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage_blocks = []
            for j in range(depths[i]):
                # 判断当前 Block 是否使用 MoE
                use_moe = (MoE_Block_inds is not None) and (j in MoE_Block_inds[i])
                if use_moe:
                    moe_cfg = dict(
                        num_experts=num_experts,
                        top_k=top_k,
                        in_channels=dims[i],
                        mid_channels=dims[i] * 4,  # FFN 中间层维度
                        pw_conv='nn.Linear',  # 点卷积的实现方式
                        act_cfg=dict(type='GELU'),  # 激活函数配置
                        noisy_gating=True,
                        use_grn=False,  # 是否使用 GRN
                        gating='cosine'  # 门控类型
                    )
                else:
                    moe_cfg = None
                
                block = ConvNeXtBlock(
                    dim=dims[i], 
                    drop_path=dp_rates[cur + j], 
                    layer_scale_init_value=layer_scale_init_value,
                    MoE_cfg=moe_cfg
                )
                stage_blocks.append(block)
            stage = nn.Sequential(*stage_blocks)
            self.stages.append(stage)
            cur += depths[i]

        self.out_indices = out_indices

        # Norm layers for output features
        norm_layer = partial(LayerNorm, eps=1e-6, data_format="channels_first")
        for i_layer in range(4):
            layer = norm_layer(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

    def init_weights(self):
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)
        else:
            super(ConvNeXt_MoE, self).init_weights()

    def forward_features(self, x):
        outs = []
        gate_losses = []  # 用于存储每个 MoE 块的 gate_loss
        for i in range(4):
            x = self.downsample_layers[i](x)
            for block in self.stages[i]:  # 遍历每个 stage 中的 block
                x, gate_loss = block(x)  # 调用 ConvNeXtBlock 的 forward 方法
                if gate_loss is not None:  # 如果存在 gate_loss，则收集
                    gate_losses.append(gate_loss)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                outs.append(x_out)
        
        # 如果有 gate_loss，则返回特征和平均 gate_loss
        if len(gate_losses) > 0:
            return outs, sum(gate_losses) / len(gate_losses)
        return outs  # 如果没有 gate_loss，则只返回特征

    def forward(self, x1, x2):
        # 提取第一个输入的特征和 gate_loss
        features1, gate_loss1 = self.forward_features(x1)
        # 提取第二个输入的特征和 gate_loss
        features2, gate_loss2 = self.forward_features(x2)

        # 合并特征
        combined_features = []
        for f1, f2 in zip(features1, features2):
            combined_features.append(torch.cat([f1, f2], dim=1))

        # 计算总的 gate_loss
        total_gate_loss = 0
        if gate_loss1 is not None:
            total_gate_loss += gate_loss1
        if gate_loss2 is not None:
            total_gate_loss += gate_loss2

        # 如果有 gate_loss，则返回特征和总 gate_loss
        if total_gate_loss > 0:
            return tuple(combined_features), total_gate_loss
        return tuple(combined_features)  # 如果没有 gate_loss，则只返回特征