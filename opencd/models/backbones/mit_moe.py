# Copyright (c) Open-CD. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from mmseg.models.utils import nlc_to_nchw
from mmseg.models.backbones import MixVisionTransformer
from mmcv.cnn import build_activation_layer
from opencd.registry import MODELS    
    
@MODELS.register_module()
class IA_MixVisionTransformer_MoE(MixVisionTransformer):
    def __init__(self, 
                 interaction_cfg=(None, None, None, None), 
                 **kwargs):
        super().__init__(**kwargs)
        assert self.num_stages == len(interaction_cfg), \
            'The length of the `interaction_cfg` should be same as the `num_stages`.'
        # cross-correlation
        self.ccs = []
        for ia_cfg in interaction_cfg:
            if ia_cfg is None:
                ia_cfg = dict(type='TwoIdentity')
            elif ia_cfg == "MoE_layer":
                # 定义稀疏MoE的配置
                ia_cfg = dict(
                    type='MoE_layer',
                    moe_cfg=dict(
                        num_experts=4,  # 专家数量
                        top_k=2,  # 每个token选择的前k个专家
                        in_channels=256,  # 输入通道数
                        mid_channels=1024,  # FFN中间层维度
                        pw_conv='nn.Linear',  # 点卷积的实现方式
                        act_cfg=dict(type='GELU'),  # 激活函数配置
                        noisy_gating=True,  # 是否使用噪声门控
                        use_grn=False,  # 是否使用GRN
                        gating='cosine'  # 门控类型
                    )
                )
            self.ccs.append(MODELS.build(ia_cfg))
        self.ccs = nn.ModuleList(self.ccs)
    
    def forward(self, x1, x2):
        outs = []
        total_gate_loss = 0  # 用于存储所有MoE模块的gate_loss

        for i, layer in enumerate(self.layers):
            x1, hw_shape = layer[0](x1)
            x2, hw_shape = layer[0](x2)
            for block in layer[1]:
                x1 = block(x1, hw_shape)
                x2 = block(x2, hw_shape)
            x1 = layer[2](x1)
            x2 = layer[2](x2)

            x1 = nlc_to_nchw(x1, hw_shape)
            x2 = nlc_to_nchw(x2, hw_shape)

            # 使用稀疏MoE模块
            if isinstance(self.ccs[i], MoE_layer):
                x1 = x1.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
                x2 = x2.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
                x1, gate_loss1 = self.ccs[i](x1)
                x2, gate_loss2 = self.ccs[i](x2)
                x1 = x1.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
                x2 = x2.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
                total_gate_loss += gate_loss1 + gate_loss2
            else:
                x1, x2 = self.ccs[i](x1, x2)

            if i in self.out_indices:
                outs.append(torch.cat([x1, x2], dim=1))

        # 如果有gate_loss，则返回输出和总gate_loss
        if total_gate_loss > 0:
            return outs, total_gate_loss
        return outs


def add_gamma_noise(x, gamma_mean=1, gamma_var_range=(1, 20)):
    """
    对输入图像 x 添加 Gamma 噪声。
    
    参数:
        x (torch.Tensor): 输入图像，shape 为 [N, C, H, W]。
        gamma_mean (float): Gamma 分布的均值，默认为 1。
        gamma_var_range (tuple): Gamma 分布的方差范围，默认为 (1, 20)。
    
    返回:
        torch.Tensor: 添加 Gamma 噪声后的图像。
    """
    # 获取输入图像的 shape
    N, C, H, W = x.shape
    
    # 随机生成 Gamma 分布的方差
    gamma_var = torch.randint(gamma_var_range[0], gamma_var_range[1] + 1, (1,)).item()
    
    # 计算 Gamma 分布的参数
    # Gamma 分布的参数化：shape = mean^2 / variance, scale = variance / mean
    shape = gamma_mean**2 / gamma_var
    scale = gamma_var / gamma_mean
    
    # 生成 Gamma 噪声，shape 为 [N, C, H, W]
    gamma_noise = torch.distributions.Gamma(shape, 1/scale).sample((N, C, H, W)).to(x.device)
    
    # 将 Gamma 噪声与输入图像相乘
    x_noisy = x * gamma_noise
    
    return x_noisy

    
@MODELS.register_module()
class IA_MixVisionTransformer_MoE_O2SP(MixVisionTransformer):
    def __init__(self, 
                 interaction_cfg=(None, None, None, None), 
                 **kwargs):
        super().__init__(**kwargs)
        assert self.num_stages == len(interaction_cfg), \
            'The length of the `interaction_cfg` should be same as the `num_stages`.'
        # cross-correlation
        self.ccs = []
        for ia_cfg in interaction_cfg:
            if ia_cfg is None:
                ia_cfg = dict(type='TwoIdentity')
            elif ia_cfg == "MoE_layer":
                # 定义稀疏MoE的配置
                ia_cfg = dict(
                    type='MoE_layer',
                    moe_cfg=dict(
                        num_experts=4,  # 专家数量
                        top_k=2,  # 每个token选择的前k个专家
                        in_channels=256,  # 输入通道数
                        mid_channels=1024,  # FFN中间层维度
                        pw_conv='nn.Linear',  # 点卷积的实现方式
                        act_cfg=dict(type='GELU'),  # 激活函数配置
                        noisy_gating=True,  # 是否使用噪声门控
                        use_grn=False,  # 是否使用GRN
                        gating='cosine'  # 门控类型
                    )
                )
            self.ccs.append(MODELS.build(ia_cfg))
        self.ccs = nn.ModuleList(self.ccs)
    
    def forward(self, x1, x2):
        x3 = add_gamma_noise(x1)
        outs = []
        total_gate_loss = 0  # 用于存储所有MoE模块的gate_loss
        total_guide_loss = 0

        for i, layer in enumerate(self.layers):
            x1, hw_shape = layer[0](x1)
            x2, hw_shape = layer[0](x2)
            x3, hw_shape = layer[0](x3)
            for block in layer[1]:
                x1 = block(x1, hw_shape)
                x2 = block(x2, hw_shape)
                x3 = block(x3, hw_shape)
            x1 = layer[2](x1)
            x2 = layer[2](x2)
            x3 = layer[2](x3)

            x1 = nlc_to_nchw(x1, hw_shape)
            x2 = nlc_to_nchw(x2, hw_shape)
            x3 = nlc_to_nchw(x3, hw_shape)

            # 使用稀疏MoE模块
            if isinstance(self.ccs[i], MoE_layer):
                x1 = x1.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
                x2 = x2.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
                x3 = x3.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
                x1, gate_loss1 = self.ccs[i](x1)
                x2, gate_loss2 = self.ccs[i](x2)
                x3, gate_loss3 = self.ccs[i](x3)
                x1 = x1.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
                x2 = x2.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
                x3 = x3.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
                total_gate_loss += gate_loss1 + gate_loss2 + gate_loss3
            else:
                x1, x2 = self.ccs[i](x1, x2)
            
            total_guide_loss += 1e-4 * (nn.L1Loss()(x1, x3)+nn.L1Loss()(x2, x3))

            if i in self.out_indices:
                outs.append(torch.cat([x1, x2], dim=1))

        # 如果有gate_loss，则返回输出和总gate_loss
        if total_guide_loss > 0:
            return outs, total_gate_loss, total_guide_loss
        return outs 
        


@MODELS.register_module()
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