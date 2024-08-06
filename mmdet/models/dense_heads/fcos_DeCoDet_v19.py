# Copyright (c) GrokCV. All rights reserved.
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmcv.cnn import ConvModule, Scale
from mmdet.registry import MODELS
from .fcos_seg_head import FCOSDepthHead
from mmcv.cnn import Conv2d, ConvModule



class CrossInvolution(nn.Module):
    def __init__(self, channels, kernel_size, num_layers=1, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.channels = channels
        self.num_layers = num_layers
        self.reduction_ratio = 4
        self.group_channels = 16
        self.groups = self.channels // self.group_channels

        self.unfold = nn.Unfold(kernel_size, stride=stride, padding=(kernel_size-1)//2)
        
        self.convs1 = nn.ModuleList([
            ConvModule(
                in_channels=channels,
                out_channels=channels // self.reduction_ratio,
                kernel_size=1,
                conv_cfg=None,
                norm_cfg=dict(type='BN'),
                act_cfg=dict(type='ReLU')) 
            for _ in range(num_layers)
        ])

        self.convs2 = nn.ModuleList([
            ConvModule(
                in_channels=channels // self.reduction_ratio,
                out_channels=self.kernel_size**2 * self.groups,
                kernel_size=1,
                stride=1,
                conv_cfg=None,
                norm_cfg=None,
                act_cfg=None) 
            for _ in range(num_layers)
        ])
    
    def forward(self, feature_map, guide_map):
        for i in range(self.num_layers):
            weight = self.convs2[i](self.convs1[i](guide_map))
            b, c, h, w = weight.shape
            weight = weight.view(b, self.groups, self.kernel_size**2, h, w)
            out = self.unfold(feature_map).view(b, self.groups, self.group_channels, self.kernel_size**2, h, w)
            out = (weight.unsqueeze(2) * out).sum(dim=3).view(b, self.channels, h, w)
            feature_map = out + feature_map  # Residual connection
        return feature_map

@MODELS.register_module()
class FCOSDeCoDetHeadV19(FCOSDepthHead):
    def __init__(self, num_layers=1, mapping=False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.feature_interaction_layers = num_layers
        self.reduction_ratio = 4
        self.group_channels = 16
        self.groups = self.feat_channels // self.group_channels
        
        self.unfold = nn.Unfold(3, 1, (3 - 1) // 2, 1)
        
        # Cross involution module
        self.cross_involution_cls = CrossInvolution(channels=self.feat_channels, kernel_size=7, num_layers=num_layers, stride=1)
        self.cross_involution_reg = CrossInvolution(channels=self.feat_channels, kernel_size=7, num_layers=num_layers, stride=1)
        
        self.mapping = mapping
        self.absolute = nn.Sigmoid()
        self.map = nn.ReLU()
        
    def _init_layers(self) -> None:
        """Initialize layers of the head."""
        super()._init_layers()
        self._init_seg_convs()
        self.conv_seg = nn.Conv2d(self.feat_channels, self.seg_out_channels, 3, padding=1)

    def _init_seg_convs(self) -> None:
        """Initialize segmentation conv layers of the head."""
        self.seg_cls_convs = nn.ModuleList()
        self.seg_reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            conv_cfg = dict(type='DCNv2') if self.dcn_on_last_conv and i == self.stacked_convs - 1 else self.conv_cfg
            self.seg_cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias
                )
            )
            self.seg_reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias
                )
            )

    def forward_single(self, x: Tensor, scale: Scale, stride: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        cls_feat, reg_feat = x, x
        seg_cls_feat, seg_reg_feat = x, x

        # Processing segmentation features
        seg_cls_feats = [seg_cls_feat]
        seg_reg_feats = [seg_reg_feat]
        
        for seg_cls_layer, seg_reg_layer in zip(self.seg_cls_convs, self.seg_reg_convs):
            seg_cls_feat = seg_cls_layer(seg_cls_feat)
            seg_reg_feat = seg_reg_layer(seg_reg_feat)
            seg_cls_feats.append(seg_cls_feat)
            seg_reg_feats.append(seg_reg_feat)

        # Processing classification features
        for i, cls_layer in enumerate(self.cls_convs):
            cls_condi_feat = self.cross_involution_cls(cls_feat, seg_cls_feats[i + 1])
            cls_feat = cls_condi_feat + cls_feat  # Added residual connection
            cls_feat = cls_layer(cls_feat)

        # Processing regression features
        for i, reg_layer in enumerate(self.reg_convs):
            reg_condi_feat = self.cross_involution_cls(reg_feat, seg_reg_feats[i + 1])
            reg_feat = reg_condi_feat + reg_feat  # Added residual connection
            reg_feat = reg_layer(reg_feat)

        # Compute segmentation score
        seg_final_feat = seg_cls_feats[-1] * seg_reg_feats[-1]
        seg_score = self.conv_seg(seg_final_feat)

        # Compute final outputs
        cls_score = self.conv_cls(cls_feat)
        bbox_pred = self.conv_reg(reg_feat)
        centerness = self.conv_centerness(reg_feat if self.centerness_on_reg else cls_feat)

        bbox_pred = scale(bbox_pred).float()
        if self.norm_on_bbox:
            bbox_pred = bbox_pred.clamp(min=0)
            if not self.training:
                bbox_pred *= stride
        else:
            bbox_pred = bbox_pred.exp()

        seg_score = self.map(seg_score) if self.mapping else self.absolute(seg_score)
        
        return cls_score, bbox_pred, centerness, seg_score