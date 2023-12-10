# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import List, Optional, Tuple

import torch
from mmcv.cnn import ConvModule, Scale, is_norm, DepthwiseSeparableConvModule
from mmdet.models import inverse_sigmoid
from mmdet.models.dense_heads import RTMDetHead
from mmdet.models.task_modules import anchor_inside_flags
from mmdet.models.utils import (filter_scores_and_topk, multi_apply,
                                select_single_mlvl, sigmoid_geometric_mean,
                                unmap)
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, cat_boxes, distance2bbox
from mmdet.utils import (ConfigType, InstanceList, OptConfigType,
                         OptInstanceList, reduce_mean)
from mmengine import ConfigDict
from mmengine.model import bias_init_with_prob, constant_init, normal_init
from mmengine.structures import InstanceData
from torch import Tensor, nn

from mmrotate.registry import MODELS, TASK_UTILS
from mmrotate.structures import RotatedBoxes, distance2obb

from .rotated_rtmdet_head import RotatedRTMDetHead



@MODELS.register_module()
class RotatedRTMDetTIDSepBNHead(RotatedRTMDetHead):
    """Rotated RTMDetHead with separated BN layers and shared conv layers.
        and deformed TSCODE
    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        share_conv (bool): Whether to share conv layers between stages.
            Defaults to True.
        scale_angle (bool): Does not support in RotatedRTMDetSepBNHead,
            Defaults to False.
        norm_cfg (:obj:`ConfigDict` or dict)): Config dict for normalization
            layer. Defaults to dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (:obj:`ConfigDict` or dict)): Config dict for activation layer.
            Defaults to dict(type='SiLU').
        pred_kernel_size (int): Kernel size of prediction layer. Defaults to 1.
        exp_on_reg (bool): Whether to apply exponential on bbox_pred.
            Defaults to False.
        add_top_out (bool): add top out, so output one more level, defaults to False.
    """
    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 share_conv: bool = True,
                 scale_angle: bool = False,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU'),
                 pred_kernel_size: int = 1,
                 exp_on_reg: bool = False,
                 use_ts_reg: bool = False,
                 upsample_cfg: ConfigType = dict(scale_factor=2, mode='nearest'),
                 **kwargs) -> None:
        self.share_conv = share_conv
        self.exp_on_reg = exp_on_reg
        assert scale_angle is False, \
            'scale_angle does not support in RotatedRTMDetTSSepBNHead'

        self.is_scale_angle = scale_angle
        self.upsample_cfg = upsample_cfg
        self.use_ts_reg = use_ts_reg

        super().__init__(
            num_classes,
            in_channels,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            pred_kernel_size=pred_kernel_size,
            scale_angle=False,
            **kwargs)

    def _init_layers(self) -> None:
        """Initialize layers of the head."""
        
        assert self.with_objectness is False

        # conv = DepthwiseSeparableConvModule if self.use_depthwise else ConvModule
        conv = ConvModule

        # TSCODE layers
        self.upsample = nn.Upsample(**self.upsample_cfg)
        self.downsamples = nn.ModuleList()
        
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()

        self.rtm_cls = nn.ModuleList()
        self.rtm_reg = nn.ModuleList()
        self.rtm_ang = nn.ModuleList()

        # TODO: set cfg as args, and this is a conv-bn-act module used in CPAN
        conv_cfg = None
        norm_cfg = dict(type='BN', momentum=0.03, eps=0.001)
        act_cfg = dict(type='Swish')
        self.downsamples.append(
            conv(
            self.in_channels,
            self.in_channels,
            3,
            stride=2,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg))
        self.downsamples.append(
            conv(
            self.in_channels,
            self.in_channels,
            3,
            stride=2,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg))
        
        assert self.in_channels == self.feat_channels, "in_channels must be equal to feat_channels."
        # stacked_convs + 1 for share channels align
        for i in range(self.stacked_convs + 1):
            # cls convs first input channel num is doubled
            chn_cls = self.in_channels * 2 if i == 0 else self.feat_channels
            chn_reg = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn_cls,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn_reg,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        # last convs 
        self.rtm_cls.append(
            nn.Conv2d(
                self.feat_channels,
                self.num_base_priors * self.cls_out_channels,
                self.pred_kernel_size,
                padding=self.pred_kernel_size // 2))
        self.rtm_reg.append(
                nn.Conv2d(
                    self.feat_channels,
                    self.num_base_priors * 4,
                    self.pred_kernel_size,
                    padding=self.pred_kernel_size // 2))
        self.rtm_ang.append(
            nn.Conv2d(
                self.feat_channels,
                self.num_base_priors * self.angle_coder.encode_size,
                self.pred_kernel_size,
                padding=self.pred_kernel_size // 2))
        
        self._init_layers02()
        
    def _init_layers02(self) -> None:
        '''
        layers 2
        layers 1 -> tscode layers
        layers 0
        '''    

        # layers 0
        self.cls_convs0 = nn.ModuleList()
        self.reg_convs0 = nn.ModuleList()

        self.rtm_cls0 = nn.ModuleList()
        self.rtm_reg0 = nn.ModuleList()
        self.rtm_ang0 = nn.ModuleList()

        for i in range(self.stacked_convs):
            chn_cls = self.in_channels if i == 0 else self.feat_channels
            chn_reg = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs0.append(
                ConvModule(
                    chn_cls,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
            self.reg_convs0.append(
                ConvModule(
                    chn_reg,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        # last convs 
        self.rtm_cls0.append(
            nn.Conv2d(
                self.feat_channels,
                self.num_base_priors * self.cls_out_channels,
                self.pred_kernel_size,
                padding=self.pred_kernel_size // 2))
        self.rtm_reg0.append(
                nn.Conv2d(
                    self.feat_channels,
                    self.num_base_priors * 4,
                    self.pred_kernel_size,
                    padding=self.pred_kernel_size // 2))
        self.rtm_ang0.append(
            nn.Conv2d(
                self.feat_channels,
                self.num_base_priors * self.angle_coder.encode_size,
                self.pred_kernel_size,
                padding=self.pred_kernel_size // 2))
        
        # layers 2 --------------------------------------------------------
        self.cls_convs2 = nn.ModuleList()
        self.reg_convs2 = nn.ModuleList()

        self.rtm_cls2 = nn.ModuleList()
        self.rtm_reg2 = nn.ModuleList()
        self.rtm_ang2 = nn.ModuleList()


        for i in range(self.stacked_convs):
            chn_cls = self.in_channels if i == 0 else self.feat_channels
            chn_reg = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs2.append(
                ConvModule(
                    chn_cls,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
            self.reg_convs2.append(
                ConvModule(
                    chn_reg,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        # share with all stack convs   
        if self.share_conv:
            for i in range(self.stacked_convs):
                self.cls_convs0[i].conv = self.cls_convs[i+1].conv
                self.reg_convs0[i].conv = self.reg_convs[i+1].conv

                self.cls_convs2[i].conv = self.cls_convs[i+1].conv
                self.reg_convs2[i].conv = self.reg_convs[i+1].conv

        # last convs 
        self.rtm_cls2.append(
            nn.Conv2d(
                self.feat_channels,
                self.num_base_priors * self.cls_out_channels,
                self.pred_kernel_size,
                padding=self.pred_kernel_size // 2))
        self.rtm_reg2.append(
            nn.Conv2d(
                self.feat_channels,
                self.num_base_priors * 4,
                self.pred_kernel_size,
                padding=self.pred_kernel_size // 2))
        self.rtm_ang2.append(
            nn.Conv2d(
                self.feat_channels,
                self.num_base_priors * self.angle_coder.encode_size,
                self.pred_kernel_size,
                padding=self.pred_kernel_size // 2))
        
    def init_weights(self) -> None:
        """Initialize weights of the head."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, mean=0, std=0.01)
            if is_norm(m):
                constant_init(m, 1)
        bias_cls = bias_init_with_prob(0.01)
        for rtm_cls, rtm_reg, rtm_ang in zip(self.rtm_cls, self.rtm_reg,
                                             self.rtm_ang):
            normal_init(rtm_cls, std=0.01, bias=bias_cls)
            normal_init(rtm_reg, std=0.01)
            normal_init(rtm_ang, std=0.01)

        if self.with_objectness:
            for rtm_obj in self.rtm_obj:
                normal_init(rtm_obj, std=0.01, bias=bias_cls)

        for rtm_cls, rtm_reg, rtm_ang in zip(self.rtm_cls0, self.rtm_reg0,
                                             self.rtm_ang0):
            normal_init(rtm_cls, std=0.01, bias=bias_cls)
            normal_init(rtm_reg, std=0.01)
            normal_init(rtm_ang, std=0.01)

        for rtm_cls, rtm_reg, rtm_ang in zip(self.rtm_cls2, self.rtm_reg2,
                                             self.rtm_ang2):
            normal_init(rtm_cls, std=0.01, bias=bias_cls)
            normal_init(rtm_reg, std=0.01)
            normal_init(rtm_ang, std=0.01)

    def forward(self, feats: Tuple[Tensor, ...]) -> tuple:
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: Usually a tuple of classification scores and bbox prediction
            - cls_scores (list[Tensor]): Classification scores for all scale
              levels, each is a 4D-tensor, the channels number is
              num_base_priors * num_classes.
            - bbox_preds (list[Tensor]): Box energies / deltas for all scale
              levels, each is a 4D-tensor, the channels number is
              num_base_priors * 4.
            - angle_preds (list[Tensor]): Angle prediction for all scale
              levels, each is a 4D-tensor, the channels number is
              num_base_priors * angle_dim.
        """
        cls_scores = []
        bbox_preds = []
        angle_preds = []
        
        assert len(feats) == 3

        p3 = feats[2]
        p2 = feats[1]
        p1 = feats[0]

        # layer0 -----------------------------------------------------------------
        cls_feat = feats[0]
        reg_feat = feats[0]
        for cls_layer in self.cls_convs0:
            cls_feat = cls_layer(cls_feat)

        for reg_layer in self.reg_convs0:
            reg_feat = reg_layer(reg_feat)
        cls_score = self.rtm_cls0[0](cls_feat)
        if self.exp_on_reg:
            reg_dist = self.rtm_reg0[0](reg_feat).exp() * self.prior_generator.strides[0][0]
        else:
            reg_dist = self.rtm_reg0[0](reg_feat) * self.prior_generator.strides[0][0]
        angle_pred = self.rtm_ang0[0](reg_feat)

        cls_scores.append(cls_score)
        bbox_preds.append(reg_dist)
        angle_preds.append(angle_pred)

        # TSCODE -----------------------------------------------------------------
        ups_p3 = self.upsample(p3)
        ups_p2 = self.upsample(p2)
        reg_feat = self.downsamples[0](ups_p2 + p1) + p2 + ups_p3

        if self.use_ts_reg:
            cls_feat = self.upsample(torch.cat((p3, self.downsamples[1](reg_feat)), 1))
        else:
            cls_feat = self.upsample(torch.cat((p3, self.downsamples[1](p2)), 1))

        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)

        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_feat)

        # print('self.prior_generator.strides', self.prior_generator.strides)

        cls_score = self.rtm_cls[0](cls_feat)
        if self.exp_on_reg:
            reg_dist = self.rtm_reg[0](reg_feat).exp() * self.prior_generator.strides[1][0]
        else:
            reg_dist = self.rtm_reg[0](reg_feat) * self.prior_generator.strides[1][0]
        angle_pred = self.rtm_ang[0](reg_feat)

        cls_scores.append(cls_score)
        bbox_preds.append(reg_dist)
        angle_preds.append(angle_pred)

        # layer2 -----------------------------------------------------------------
        cls_feat = feats[2]
        reg_feat = feats[2]

        for cls_layer in self.cls_convs2:
            cls_feat = cls_layer(cls_feat)

        for reg_layer in self.reg_convs2:
            reg_feat = reg_layer(reg_feat)

        cls_score = self.rtm_cls2[0](cls_feat)
        if self.exp_on_reg:
            reg_dist = self.rtm_reg2[0](reg_feat).exp() * self.prior_generator.strides[2][0]
        else:
            reg_dist = self.rtm_reg2[0](reg_feat) * self.prior_generator.strides[2][0]
        angle_pred = self.rtm_ang2[0](reg_feat)

        cls_scores.append(cls_score)
        bbox_preds.append(reg_dist)
        angle_preds.append(angle_pred)

        return tuple(cls_scores), tuple(bbox_preds), tuple(angle_preds)
    
  