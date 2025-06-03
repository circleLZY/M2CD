# Copyright (c) Open-CD. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from mmseg.models.backbones import ResNet
from mmcv.cnn import build_activation_layer
from opencd.registry import MODELS

@MODELS.register_module()
class IA_ResNet_MoE(ResNet):
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
            elif ia_cfg == "MoE_layer_2":
                # Define sparse MoE configuration
                in_channels = kwargs.get('base_channels', 64) * 2**(len(self.ccs))
                ia_cfg = dict(
                    type='MoE_layer_2',
                    moe_cfg=dict(
                        num_experts=4,  # Number of experts
                        top_k=2,  # Top-k experts to select for each token
                        in_channels=in_channels,  # Input channels
                        mid_channels=in_channels * 4,  # Middle layer dimension
                        pw_conv='nn.Linear',  # Point-wise convolution implementation
                        act_cfg=dict(type='GELU'),  # Activation function
                        noisy_gating=True,  # Whether to use noisy gating
                        use_grn=False,  # Whether to use GRN
                        gating='cosine'  # Gating type
                    )
                )
            self.ccs.append(MODELS.build(ia_cfg))
        self.ccs = nn.ModuleList(self.ccs)
    
    def forward(self, x1, x2):
        """Forward function."""
        def _stem_forward(x):
            if self.deep_stem:
                x = self.stem(x)
            else:
                x = self.conv1(x)
                x = self.norm1(x)
                x = self.relu(x)
            x = self.maxpool(x)
            return x
            
        x1 = _stem_forward(x1)
        x2 = _stem_forward(x2)
        outs = []
        total_gate_loss = 0  # For storing all MoE modules' gate losses
        
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x1 = res_layer(x1)
            x2 = res_layer(x2)
            
            # Use sparse MoE module
            if isinstance(self.ccs[i], MoE_layer_2):
                # Convert from NCHW to NHWC format for MoE
                x1_shape = x1.shape
                x2_shape = x2.shape
                
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
                
        # If there's gate_loss, return outputs and total gate_loss
        if total_gate_loss > 0:
            return tuple(outs), total_gate_loss
        return tuple(outs)


def add_gamma_noise(x, gamma_mean=1, gamma_var_range=(1, 20)):
    """
    Add Gamma noise to the input image x.
    
    Args:
        x (torch.Tensor): Input image, shape [N, C, H, W].
        gamma_mean (float): Mean of Gamma distribution, default is 1.
        gamma_var_range (tuple): Variance range of Gamma distribution, default is (1, 20).
    
    Returns:
        torch.Tensor: Image with added Gamma noise.
    """
    # Get shape of input image
    N, C, H, W = x.shape
    
    # Randomly generate variance for Gamma distribution
    gamma_var = torch.randint(gamma_var_range[0], gamma_var_range[1] + 1, (1,)).item()
    
    # Calculate parameters for Gamma distribution
    # Parameterization: shape = mean^2 / variance, scale = variance / mean
    shape = gamma_mean**2 / gamma_var
    scale = gamma_var / gamma_mean
    
    # Generate Gamma noise, shape [N, C, H, W]
    gamma_noise = torch.distributions.Gamma(shape, 1/scale).sample((N, C, H, W)).to(x.device)
    
    # Multiply Gamma noise with input image
    x_noisy = x * gamma_noise
    
    return x_noisy


@MODELS.register_module()
class IA_ResNet_MoE_O2SP(ResNet):
    """Interaction ResNet backbone with Mixture of Experts and One-to-Some Perturbation (O2SP).
    
    Args:
        interaction_cfg (Sequence[dict]): Interaction strategies for the stages.
            The length should be the same as `num_stages`.
            Default: (None, None, None, None).
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        in_channels (int): Number of input image channels. Default: 3.
        stem_channels (int): Number of stem channels. Default: 64.
        base_channels (int): Number of base channels of res layer. Default: 64.
        num_stages (int): Resnet stages, normally 4. Default: 4.
        strides (Sequence[int]): Strides of the first block of each stage.
            Default: (1, 2, 2, 2).
        dilations (Sequence[int]): Dilation of each stage.
            Default: (1, 1, 1, 1).
        out_indices (Sequence[int]): Output from which stages.
            Default: (0, 1, 2, 3).
        style (str): `pytorch` or `caffe`. Default: 'pytorch'.
        deep_stem (bool): Replace 7x7 conv in input stem with 3 3x3 conv.
            Default: False.
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Default: False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Default: -1.
        conv_cfg (dict | None): Dictionary to construct and config conv layer.
            Default: None.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: dict(type='BN', requires_grad=True).
        norm_eval (bool): Whether to set norm layers to eval mode. Default: False.
        dcn (dict | None): Dictionary to construct and config DCN conv layer.
            Default: None.
        stage_with_dcn (Sequence[bool]): Whether to set DCN conv for each
            stage. Default: (False, False, False, False).
        plugins (list[dict]): List of plugins for stages. Default: None.
        multi_grid (Sequence[int]|None): Multi grid dilation rates of last
            stage. Default: None.
        contract_dilation (bool): Whether contract first dilation of each layer
            Default: False.
        with_cp (bool): Use checkpoint or not. Default: False.
        zero_init_residual (bool): Whether to use zero init for last norm layer
            in resblocks to let them behave as identity. Default: True.
        pretrained (str, optional): model pretrained path. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """
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
            elif ia_cfg == "MoE_layer_2":
                # Define sparse MoE configuration
                in_channels = kwargs.get('base_channels', 64) * 2**(len(self.ccs))
                ia_cfg = dict(
                    type='MoE_layer_2',
                    moe_cfg=dict(
                        num_experts=4,  # Number of experts
                        top_k=2,  # Top-k experts to select for each token
                        in_channels=in_channels,  # Input channels
                        mid_channels=in_channels * 4,  # Middle layer dimension
                        pw_conv='nn.Linear',  # Point-wise convolution implementation
                        act_cfg=dict(type='GELU'),  # Activation function
                        noisy_gating=True,  # Whether to use noisy gating
                        use_grn=False,  # Whether to use GRN
                        gating='cosine'  # Gating type
                    )
                )
            self.ccs.append(MODELS.build(ia_cfg))
        self.ccs = nn.ModuleList(self.ccs)
    
    def forward(self, x1, x2):
        """Forward function."""
        def _stem_forward(x):
            if self.deep_stem:
                x = self.stem(x)
            else:
                x = self.conv1(x)
                x = self.norm1(x)
                x = self.relu(x)
            x = self.maxpool(x)
            return x
        
        # Create perturbed version of x1 with Gamma noise
        x3 = add_gamma_noise(x1)
        
        x1 = _stem_forward(x1)
        x2 = _stem_forward(x2)
        x3 = _stem_forward(x3)
        
        outs = []
        total_gate_loss = 0  # For storing all MoE modules' gate losses
        total_guide_loss = 0  # For storing O2SP guide losses
        
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x1 = res_layer(x1)
            x2 = res_layer(x2)
            x3 = res_layer(x3)
            
            # Use sparse MoE module
            if isinstance(self.ccs[i], MoE_layer_2):
                # Convert from NCHW to NHWC format for MoE
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
            
            # Calculate guide loss for O2SP
            total_guide_loss += 1e-4 * (nn.L1Loss()(x1, x3) + nn.L1Loss()(x2, x3))
            
            if i in self.out_indices:
                outs.append(torch.cat([x1, x2], dim=1))
        
        # If there's gate_loss and guide_loss, return outputs, total_gate_loss and total_guide_loss
        if total_guide_loss > 0:
            return tuple(outs), total_gate_loss, total_guide_loss
        return tuple(outs)


@MODELS.register_module()
class IA_ResNetV1c_MoE(IA_ResNet_MoE):
    """ResNetV1c variant with Mixture of Experts.
    
    Compared with default ResNet(ResNetV1b), ResNetV1c replaces the 7x7 conv in
    the input stem with three 3x3 convs.
    """
    def __init__(self, **kwargs):
        super(IA_ResNetV1c_MoE, self).__init__(
            deep_stem=True, avg_down=False, **kwargs)


@MODELS.register_module()
class IA_ResNetV1c_MoE_O2SP(IA_ResNet_MoE_O2SP):
    """ResNetV1c variant with Mixture of Experts and One-to-Some Perturbation (O2SP).
    
    Compared with default ResNet(ResNetV1b), ResNetV1c replaces the 7x7 conv in
    the input stem with three 3x3 convs.
    """
    def __init__(self, **kwargs):
        super(IA_ResNetV1c_MoE_O2SP, self).__init__(
            deep_stem=True, avg_down=False, **kwargs)


# Define MoE_layer class if not already defined elsewhere
@MODELS.register_module()
class MoE_layer_2(nn.Module):
    def __init__(self, moe_cfg):
        super(MoE_layer_2, self).__init__()
        self.noisy_gating = moe_cfg['noisy_gating']
        self.num_experts = moe_cfg['num_experts']
        self.input_size = moe_cfg['in_channels']
        self.k = moe_cfg['top_k']
        
        # Dynamically load pw_conv
        if isinstance(moe_cfg['pw_conv'], str):
            moe_cfg['pw_conv'] = eval(moe_cfg['pw_conv'])
        
        # Instantiate experts
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
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=-1)  
        
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
        x = x.reshape(-1, x.shape[-1])
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
        return y, loss


class SparseDispatcher(object):
    def __init__(self, num_experts, gates):
        self._gates = gates
        self._num_experts = num_experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = sorted_experts[index_sorted_experts[:, 1], 0]
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
        # x shape is (batch_size * height * width, model_dim)
        cosine_projector = self.cosine_projector
        sim_matrix = self.sim_matrix
        logits = torch.matmul(F.normalize(cosine_projector(x), dim=1),
                             F.normalize(sim_matrix, dim=0))
        logit_scale = torch.clamp(self.temperature, max=self.clamp_max).exp()
        logits = logits * logit_scale
        return logits