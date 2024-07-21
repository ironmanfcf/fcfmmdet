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



@MODELS.register_module()
class FCOSDeCoDetHeadV16(FCOSDepthHead):
    def __init__(self, num_layers=1, mapping=False,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.feature_interaction_layers = num_layers
        self.reduction_ratio = 4
        self.group_channels = 16
        self.groups = self.feat_channels // self.group_channels
        
        # 各种卷积层定义

        self.unfold = nn.Unfold(3, 1, (3 - 1) // 2, 1)
        
        # 定义 involution 模块
        # self.inner_involution = involution(channels=self.feat_channels, kernel_size=7, stride=1)
       
        # 定义 involution 模块
        self.cross_involution_cls = cross_involution(channels=self.feat_channels, kernel_size=7, num_layers=num_layers, stride=1)
        self.cross_involution_reg = cross_involution(channels=self.feat_channels, kernel_size=7, num_layers=num_layers, stride=1)
        
        self.mapping=mapping
        self.absolute = nn.Sigmoid()
        self.mapping = nn.ReLU()
        

    def _init_layers(self) -> None:
        super()._init_layers()
 
    def forward_single(self, x: Tensor, scale: Scale, stride: int) -> Tuple[Tensor, Tensor, Tensor]:
        cls_feat = x
        reg_feat = x
        seg_feat = x

        seg_feats = [seg_feat]  # 初始化保存 seg_feat 特征的列表

        # 处理分割特征
        for seg_layer in self.seg_convs:
            seg_feat = seg_layer(seg_feat)
            seg_feats.append(seg_feat)  # 每次经过 seg_layer 后保存 seg_feat 进入列表

        # 处理分类特征
        for i, cls_layer in enumerate(self.cls_convs):
            cls_feat_condi = self.cross_involution_cls(cls_feat, seg_feats[i+1])  # 用seg_feats进行调制
            cls_feat = cls_feat + cls_feat_condi  # 恒等映射，可以在此处添加其他操作
            cls_feat = cls_layer(cls_feat)
        # for cls_layer in self.cls_convs:
        #     cls_feat = cls_layer(cls_feat)


        # # 处理回归特征
        for i, reg_layer in enumerate(self.reg_convs):
            reg_feat_condi  = self.cross_involution_reg(reg_feat, seg_feats[i+1])  # 用seg_feats进行调制
            reg_feat = reg_feat_condi + reg_feat  # 恒等映射，可以在此处添加其他操作
            reg_feat = reg_layer(reg_feat)
            
        # 处理回归特征
        # for reg_layer in self.reg_convs:
        #     reg_feat = reg_layer(reg_feat)

        # 使用最后一层 seg_feat 进行分割映射
        seg_score = self.conv_seg(seg_feats[-1])

        # 计算最终输出
        cls_score = self.conv_cls(cls_feat)
        bbox_pred = self.conv_reg(reg_feat)

        if self.centerness_on_reg:
            centerness = self.conv_centerness(reg_feat)
        else:
            centerness = self.conv_centerness(cls_feat)
        
        bbox_pred = scale(bbox_pred).float()
        if self.norm_on_bbox:
            bbox_pred = bbox_pred.clamp(min=0)
            if not self.training:
                bbox_pred *= stride
        else:
            bbox_pred = bbox_pred.exp()

        if self.mapping:
            seg_score = self.mapping(seg_score)
        else: 
            seg_score = self.absolute(seg_score)

        return cls_score, bbox_pred, centerness, seg_score
    
    

class cross_involution(nn.Module):
    def __init__(self, channels, kernel_size, num_layers=1, stride=1):
        super(cross_involution, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.channels = channels
        self.num_layers = num_layers
        self.reduction_ratio = 4
        self.group_channels = 16
        self.groups = self.channels // self.group_channels
        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()

        for _ in range(num_layers):
            self.convs1.append(ConvModule(
                in_channels=channels,
                out_channels=channels // self.reduction_ratio,
                kernel_size=1,
                conv_cfg=None,
                norm_cfg=dict(type='BN'),
                act_cfg=dict(type='ReLU')))
            self.convs2.append(ConvModule(
                in_channels=channels // self.reduction_ratio,
                out_channels=self.kernel_size**2 * self.groups,
                kernel_size=1,
                stride=1,
                conv_cfg=None,
                norm_cfg=None,
                act_cfg=None))

        self.unfold = nn.Unfold(kernel_size, 1, (kernel_size-1)//2, stride)

    def forward(self, feature_map, guide_map):
        for i in range(self.num_layers):
            weight = self.convs2[i](self.convs1[i](guide_map))
            b, c, h, w = weight.shape
            weight = weight.view(b, self.groups, self.kernel_size**2, h, w).unsqueeze(2)
            out = self.unfold(feature_map).view(b, self.groups, self.group_channels, self.kernel_size**2, h, w)
            out = (weight * out).sum(dim=3).view(b, self.channels, h, w)
            feature_map = out + feature_map  # residual connection
        return feature_map