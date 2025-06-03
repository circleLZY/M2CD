# Copyright (c) Open-CD. All rights reserved.
from typing import List, Optional

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from opencd.registry import MODELS
from mmseg.models.segmentors.base import BaseSegmentor
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)

from .siamencoder_decoder import SiamEncoderDecoder


@MODELS.register_module()
class DIEncoderDecoder(SiamEncoderDecoder):
    """Dual Input Encoder Decoder segmentors.

    DIEncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """
    
    def extract_feat(self, inputs: Tensor) -> List[Tensor]:
        """Extract features from images."""
        # `in_channels` is not in the ATTRIBUTE for some backbone CLASS.
        img_from, img_to = torch.split(inputs, self.backbone_inchannels, dim=1)
        x = self.backbone(img_from, img_to)
        if self.with_neck:
            x = self.neck(x)
        return x


@MODELS.register_module()
class DistillDIEncoderDecoder_S(DIEncoderDecoder):
    def __init__(self,
                #  teacher_l: DIEncoderDecoder,  # 教师模型
                #  teacher_m: DIEncoderDecoder,  # 教师模型
                #  teacher_s: DIEncoderDecoder,  # 教师模型
                 distill_loss,               # 蒸馏损失函数配置
                 backbone: ConfigType,
                 decode_head: ConfigType,
                 neck: OptConfigType = None,
                 auxiliary_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 pretrained: Optional[str] = None,
                 init_cfg: OptMultiConfig = None,
                 backbone_inchannels: int = 3,
                 
                 init_cfg_t_l: OptMultiConfig = None,
                 init_cfg_t_m: OptMultiConfig = None,
                 init_cfg_t_s: OptMultiConfig = None,
                 ):
        # 调用父类 DIEncoderDecoder 的初始化
        super().__init__(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            pretrained=pretrained,
            init_cfg=init_cfg,
            backbone_inchannels=backbone_inchannels
        )
        
        # self.teacher_l = MODELS.build(teacher_l)
        # self.teacher_m = MODELS.build(teacher_m)
        # self.teacher_s = MODELS.build(teacher_s)
        self.teacher_l = DIEncoderDecoder(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            pretrained=pretrained,
            init_cfg=init_cfg_t_l,
            backbone_inchannels=backbone_inchannels
        )
        self.teacher_m = DIEncoderDecoder(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            pretrained=pretrained,
            init_cfg=init_cfg_t_m,
            backbone_inchannels=backbone_inchannels
        )
        self.teacher_s = DIEncoderDecoder(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            pretrained=pretrained,
            init_cfg=init_cfg_t_s,
            backbone_inchannels=backbone_inchannels
        )
        
        # 确保教师模型不参与参数更新
        # for param in self.teacher_l.parameters():
        #     param.requires_grad = False
        # for param in self.teacher_m.parameters():
        #     param.requires_grad = False
        # for param in self.teacher_s.parameters():
        #     param.requires_grad = False
            
        # 构建蒸馏损失函数
        self.distill_loss = MODELS.build(distill_loss)


    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        # 经典交叉熵等损失
        x_s = self.extract_feat(inputs)
        
        losses = dict()
        loss_decode = self._decode_head_forward_train(x_s, data_samples)
        losses.update(loss_decode)
        
        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(x_s, data_samples)
            losses.update(loss_aux)

        # 计算蒸馏损失
        student_output = self.decode_head.forward(x_s)  # 学生模型的输出

        # 初始化教师输出
        teacher_outputs = []

        self.teacher_l.eval()
        self.teacher_m.eval()
        self.teacher_s.eval()
        for i, data_sample in enumerate(data_samples):
            gt_seg = data_sample.gt_sem_seg.data  # 假设每个data_sample包含 ground truth
            change_area_ratio = (gt_seg > 0).float().mean()  # 计算变化区域比例
            
            # 根据变化区域比例选择教师模型并计算输出
            with torch.no_grad():  # 确保教师模型不更新梯度
                if change_area_ratio < 0.05:
                    teacher_output = self.teacher_s.decode_head.forward(self.teacher_s.extract_feat(inputs[i:i+1]))
                elif change_area_ratio < 0.2:
                    teacher_output = self.teacher_m.decode_head.forward(self.teacher_m.extract_feat(inputs[i:i+1]))
                else:
                    teacher_output = self.teacher_l.decode_head.forward(self.teacher_l.extract_feat(inputs[i:i+1]))

            teacher_outputs.append(teacher_output)

        # 将教师输出堆叠成 (N, C, H, W) 的张量
        teacher_outputs = torch.cat(teacher_outputs, dim=0)

        # 计算学生与教师模型输出之间的蒸馏损失
        loss_distill = dict()
        loss_distill['distill'] = self.distill_loss(student_output, teacher_outputs)
        losses.update(loss_distill)
        
        
        return losses



@MODELS.register_module()
class DistillDIEncoderDecoder_S_TwoTeachers(DIEncoderDecoder):
    def __init__(self,
                 distill_loss,               # 蒸馏损失函数配置
                 backbone: ConfigType,
                 decode_head: ConfigType,
                 neck: OptConfigType = None,
                 auxiliary_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 pretrained: Optional[str] = None,
                 init_cfg: OptMultiConfig = None,
                 backbone_inchannels: int = 3,
                 
                 init_cfg_t_l: OptMultiConfig = None,
                 init_cfg_t_s: OptMultiConfig = None,
                 ):
        # 调用父类 DIEncoderDecoder 的初始化
        super().__init__(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            pretrained=pretrained,
            init_cfg=init_cfg,
            backbone_inchannels=backbone_inchannels
        )
        
        self.teacher_l = DIEncoderDecoder(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            pretrained=pretrained,
            init_cfg=init_cfg_t_l,
            backbone_inchannels=backbone_inchannels
        )

        self.teacher_s = DIEncoderDecoder(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            pretrained=pretrained,
            init_cfg=init_cfg_t_s,
            backbone_inchannels=backbone_inchannels
        )
            
        # 构建蒸馏损失函数
        self.distill_loss = MODELS.build(distill_loss)


    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        # 经典交叉熵等损失
        x_s = self.extract_feat(inputs)
        
        losses = dict()
        loss_decode = self._decode_head_forward_train(x_s, data_samples)
        losses.update(loss_decode)
        
        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(x_s, data_samples)
            losses.update(loss_aux)

        # 计算蒸馏损失
        student_output = self.decode_head.forward(x_s)  # 学生模型的输出

        # 初始化教师输出
        teacher_outputs = []

        for i, data_sample in enumerate(data_samples):
            gt_seg = data_sample.gt_sem_seg.data  # 假设每个data_sample包含 ground truth
            change_area_ratio = (gt_seg > 0).float().mean()  # 计算变化区域比例

            # 根据变化区域比例选择教师模型并计算输出
            with torch.no_grad():  # 确保教师模型不更新梯度
                if change_area_ratio < 0.10:
                    teacher_output = self.teacher_s.decode_head.forward(self.teacher_s.extract_feat(inputs[i:i+1]))
                else:
                    teacher_output = self.teacher_l.decode_head.forward(self.teacher_l.extract_feat(inputs[i:i+1]))

            teacher_outputs.append(teacher_output)

        # 将教师输出堆叠成 (N, C, H, W) 的张量
        teacher_outputs = torch.cat(teacher_outputs, dim=0)

        # 计算学生与教师模型输出之间的蒸馏损失
        loss_distill = dict()
        loss_distill['distill'] = self.distill_loss(student_output, teacher_outputs)
        losses.update(loss_distill)
        
        return losses


@MODELS.register_module()
class DIEncoderDecoder_MoE(SiamEncoderDecoder):
    """SiamEncoderDecoder with MoE (Mixture of Experts) support.

    This class extends SiamEncoderDecoder to handle the additional gate_loss
    returned by the MoE-enabled backbone.
    """

    def extract_feat(self, inputs: Tensor):
        """Extract features from images and return gate_loss from MoE.

        Args:
            inputs (Tensor): Input tensor with shape (N, C, H, W).

        Returns:
            Tuple[List[Tensor], Tensor]: A tuple containing:
                - List of feature maps from the backbone.
                - Gate loss from the MoE modules.
        """
        # Split input into two images (from and to)
        img_from, img_to = torch.split(inputs, self.backbone_inchannels, dim=1)
        
        # Extract features and gate_loss from the backbone
        features, gate_loss = self.backbone(img_from, img_to)
        
        # Apply neck if exists
        if self.with_neck:
            features = self.neck(features)
        
        return features, gate_loss

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (Tensor): Input images.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: A dictionary of loss components, including:
                - Main segmentation loss.
                - Auxiliary loss (if applicable).
                - Gate loss from MoE modules.
        """
        # Extract features and gate_loss
        features, gate_loss = self.extract_feat(inputs)
        
        # Calculate main segmentation loss
        losses = dict()
        loss_decode = self._decode_head_forward_train(features, data_samples)
        losses.update(loss_decode)
        
        # Calculate auxiliary loss (if applicable)
        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(features, data_samples)
            losses.update(loss_aux)
        
        # Add gate_loss to the loss dictionary
        if gate_loss is not None:
            losses['gate_loss'] = gate_loss
        
        return losses

    def _forward(self, inputs: Tensor, data_samples: OptSampleList = None) -> Tensor:
        """Network forward process.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and `gt_sem_seg`.

        Returns:
            Tensor: Forward output of model without any post-processes.
        """
        # Extract features and gate_loss
        features, _ = self.extract_feat(inputs)
        
        # Forward through decode head
        return self.decode_head.forward(features)

    def encode_decode(self, inputs: Tensor, batch_img_metas: List[dict]) -> Tensor:
        """Encode images with backbone and decode into a semantic segmentation map.

        Args:
            inputs (Tensor): Input tensor with shape (N, C, H, W).
            batch_img_metas (List[dict]): List of image metainfo.

        Returns:
            Tensor: Segmentation logits with shape (N, num_classes, H, W).
        """
        # Extract features and gate_loss
        features, _ = self.extract_feat(inputs)
        
        # Predict segmentation logits
        seg_logits = self.decode_head.predict(features, batch_img_metas, self.test_cfg)
        return seg_logits



@MODELS.register_module()
class DIEncoderDecoder_MoE_O2SP(SiamEncoderDecoder):
    """SiamEncoderDecoder with MoE (Mixture of Experts) support.

    This class extends SiamEncoderDecoder to handle the additional gate_loss
    returned by the MoE-enabled backbone.
    """

    def extract_feat(self, inputs: Tensor):
        """Extract features from images and return gate_loss from MoE.

        Args:
            inputs (Tensor): Input tensor with shape (N, C, H, W).

        Returns:
            Tuple[List[Tensor], Tensor]: A tuple containing:
                - List of feature maps from the backbone.
                - Gate loss from the MoE modules.
        """
        # Split input into two images (from and to)
        img_from, img_to = torch.split(inputs, self.backbone_inchannels, dim=1)
        
        # Extract features and gate_loss from the backbone
        features, gate_loss, guide_loss = self.backbone(img_from, img_to)
        
        # Apply neck if exists
        if self.with_neck:
            features = self.neck(features)
        
        return features, gate_loss, guide_loss

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (Tensor): Input images.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: A dictionary of loss components, including:
                - Main segmentation loss.
                - Auxiliary loss (if applicable).
                - Gate loss from MoE modules.
        """
        # Extract features and gate_loss
        features, gate_loss, guide_loss = self.extract_feat(inputs)
        
        # Calculate main segmentation loss
        losses = dict()
        loss_decode = self._decode_head_forward_train(features, data_samples)
        losses.update(loss_decode)
        
        # Calculate auxiliary loss (if applicable)
        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(features, data_samples)
            losses.update(loss_aux)
        
        # Add gate_loss to the loss dictionary
        if gate_loss is not None:
            losses['gate_loss'] = gate_loss
            
        if guide_loss is not None:
            losses['guide_loss'] = guide_loss
        
        return losses

    def _forward(self, inputs: Tensor, data_samples: OptSampleList = None) -> Tensor:
        """Network forward process.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and `gt_sem_seg`.

        Returns:
            Tensor: Forward output of model without any post-processes.
        """
        # Extract features and gate_loss
        features, _, _ = self.extract_feat(inputs)
        
        # Forward through decode head
        return self.decode_head.forward(features)

    def encode_decode(self, inputs: Tensor, batch_img_metas: List[dict]) -> Tensor:
        """Encode images with backbone and decode into a semantic segmentation map.

        Args:
            inputs (Tensor): Input tensor with shape (N, C, H, W).
            batch_img_metas (List[dict]): List of image metainfo.

        Returns:
            Tensor: Segmentation logits with shape (N, num_classes, H, W).
        """
        # Extract features and gate_loss
        features, _, _ = self.extract_feat(inputs)
        
        # Predict segmentation logits
        seg_logits = self.decode_head.predict(features, batch_img_metas, self.test_cfg)
        return seg_logits


@MODELS.register_module()
class DIEncoderDecoderDistill_v2(DIEncoderDecoder):
    def __init__(self,
                 distill_loss,               # 蒸馏损失函数配置
                 backbone: ConfigType,
                 decode_head: ConfigType,
                 neck: OptConfigType = None,
                 auxiliary_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 pretrained: Optional[str] = None,
                 init_cfg: OptMultiConfig = None,
                 backbone_inchannels: int = 3,
                 
                 init_cfg_t_intact: OptMultiConfig = None,
                 init_cfg_t_damaged: OptMultiConfig = None,
                 init_cfg_t_destroyed: OptMultiConfig = None,
                 **kwargs):
        # 调用父类 DIEncoderDecoder 的初始化
        super().__init__(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            pretrained=pretrained,
            init_cfg=init_cfg,
            backbone_inchannels=backbone_inchannels
        )
        
        # Build teachers
        self.teacher_intact = DIEncoderDecoder(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            pretrained=pretrained,
            init_cfg=init_cfg_t_intact,
            backbone_inchannels=backbone_inchannels
        )
        self.teacher_damaged = DIEncoderDecoder(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            pretrained=pretrained,
            init_cfg=init_cfg_t_damaged,
            backbone_inchannels=backbone_inchannels
        )
        self.teacher_destroyed = DIEncoderDecoder(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            pretrained=pretrained,
            init_cfg=init_cfg_t_destroyed,
            backbone_inchannels=backbone_inchannels
        )
        
        # Freeze teachers
        for teacher in [self.teacher_intact, self.teacher_damaged, self.teacher_destroyed]:
            if teacher is not None:
                for param in teacher.parameters():
                    param.requires_grad = False
        
        # Distill loss
        self.distill_loss = MODELS.build(distill_loss)


    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        """多教师蒸馏损失计算"""
        # 1. 常规分割损失（学生模型）
        x_s = self.extract_feat(inputs)
        losses = dict()
        
        # 原始分割损失保持不变
        loss_decode = self._decode_head_forward_train(x_s, data_samples)
        losses.update(loss_decode)
        
        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(x_s, data_samples)
            losses.update(loss_aux)

        # 2. 多教师蒸馏损失
        batch_size = inputs.size(0)
        num_teachers = 3  # intact/damaged/destroyed
        teacher_outputs = []
        
        # 将输入拆分为各教师数据 (假设输入形状为 [B*3, C, H, W])
        split_inputs = torch.chunk(inputs, num_teachers, dim=0)
        
        with torch.no_grad():
            # 处理intact教师数据
            intact_output = self.teacher_intact.decode_head.forward(
                self.teacher_intact.extract_feat(split_inputs[0]))
            teacher_outputs.append(intact_output)
            
            # 处理damaged教师数据
            damaged_output = self.teacher_damaged.decode_head.forward(
                self.teacher_damaged.extract_feat(split_inputs[1]))
            teacher_outputs.append(damaged_output)
            
            # 处理destroyed教师数据
            destroyed_output = self.teacher_destroyed.decode_head.forward(
                self.teacher_destroyed.extract_feat(split_inputs[2]))
            teacher_outputs.append(destroyed_output)
        
        # 合并教师输出 [3*B, C, H, W]
        teacher_outputs = torch.cat(teacher_outputs, dim=0)
        
        # 学生模型输出（已经是处理过所有教师数据的特征）
        student_output = self.decode_head.forward(x_s)
        
        # 计算蒸馏损失
        losses['loss_distill'] = self.distill_loss(
            student_output,  # [3*B, C, H, W]
            teacher_outputs  # [3*B, C, H, W]
        )
        
        return losses



@MODELS.register_module()
class DIEncoderDecoderDistill_v2_2T(DIEncoderDecoder):
    def __init__(self,
                 distill_loss,               # 蒸馏损失函数配置
                 backbone: ConfigType,
                 decode_head: ConfigType,
                 neck: OptConfigType = None,
                 auxiliary_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 pretrained: Optional[str] = None,
                 init_cfg: OptMultiConfig = None,
                 backbone_inchannels: int = 3,
                 
                 init_cfg_t_intact: OptMultiConfig = None,
                 init_cfg_t_damaged_destroyed: OptMultiConfig = None,
                 **kwargs):
        # 调用父类 DIEncoderDecoder 的初始化
        super().__init__(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            pretrained=pretrained,
            init_cfg=init_cfg,
            backbone_inchannels=backbone_inchannels
        )
        
        # Build teachers
        self.teacher_intact = DIEncoderDecoder(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            pretrained=pretrained,
            init_cfg=init_cfg_t_intact,
            backbone_inchannels=backbone_inchannels
        )
        self.teacher_damaged_destroyed = DIEncoderDecoder(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            pretrained=pretrained,
            init_cfg=init_cfg_t_damaged_destroyed,
            backbone_inchannels=backbone_inchannels
        )
        
        # Freeze teachers
        for teacher in [self.teacher_intact, self.teacher_damaged_destroyed]:
            if teacher is not None:
                for param in teacher.parameters():
                    param.requires_grad = False
        
        # Distill loss
        self.distill_loss = MODELS.build(distill_loss)


    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        """多教师蒸馏损失计算"""
        # 1. 常规分割损失（学生模型）
        x_s = self.extract_feat(inputs)
        losses = dict()
        
        # 原始分割损失保持不变
        loss_decode = self._decode_head_forward_train(x_s, data_samples)
        losses.update(loss_decode)
        
        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(x_s, data_samples)
            losses.update(loss_aux)

        # 2. 多教师蒸馏损失
        batch_size = inputs.size(0)
        num_teachers = 2  # intact/damaged/destroyed
        teacher_outputs = []
        
        # 将输入拆分为各教师数据 (假设输入形状为 [B*3, C, H, W])
        split_inputs = torch.chunk(inputs, num_teachers, dim=0)
        
        with torch.no_grad():
            # 处理intact教师数据
            intact_output = self.teacher_intact.decode_head.forward(
                self.teacher_intact.extract_feat(split_inputs[0]))
            teacher_outputs.append(intact_output)
            
            # 处理damaged_destroyed教师数据
            destroyed_output = self.teacher_damaged_destroyed.decode_head.forward(
                self.teacher_damaged_destroyed.extract_feat(split_inputs[1]))
            teacher_outputs.append(destroyed_output)
        
        # 合并教师输出 [3*B, C, H, W]
        teacher_outputs = torch.cat(teacher_outputs, dim=0)
        
        # 学生模型输出（已经是处理过所有教师数据的特征）
        student_output = self.decode_head.forward(x_s)
        
        # 计算蒸馏损失
        losses['loss_distill'] = self.distill_loss(
            student_output,  # [3*B, C, H, W]
            teacher_outputs  # [3*B, C, H, W]
        )
        
        return losses