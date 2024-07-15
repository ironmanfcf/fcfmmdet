# Copyright (c) GrokCV. All rights reserved.
import copy
from abc import ABCMeta, abstractmethod
from inspect import signature
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmcv.cnn import ConvModule, Scale
from .fcos_seg_head import FCOSDepthHead


from mmdet.registry import MODELS

@MODELS.register_module()
class FCOSDeCoDetHeadV3(FCOSDepthHead):
    """Anchor-free head used in `FCOS <https://arxiv.org/abs/1904.01355>`_.

    The FCOS head does not use anchor boxes. Instead bounding boxes are
    predicted at each pixel and a centerness measure is used to suppress
    low-quality predictions.
    Here norm_on_bbox, centerness_on_reg, dcn_on_last_conv are training
    tricks used in official repo, which will bring remarkable mAP gains
    of up to 4.9. Please see https://github.com/tianzhi0549/FCOS for
    more detail.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        strides (Sequence[int] or Sequence[Tuple[int, int]]): Strides of points
            in multiple feature levels. Defaults to (4, 8, 16, 32, 64).
        regress_ranges (Sequence[Tuple[int, int]]): Regress range of multiple
            level points.
        center_sampling (bool): If true, use center sampling.
            Defaults to False.
        center_sample_radius (float): Radius of center sampling.
            Defaults to 1.5.
        norm_on_bbox (bool): If true, normalize the regression targets with
            FPN strides. Defaults to False.
        centerness_on_reg (bool): If true, position centerness on the
            regress branch. Please refer to https://github.com/tianzhi0549/FCOS/issues/89#issuecomment-516877042.
            Defaults to False.
        conv_bias (bool or str): If specified as `auto`, it will be decided by
            the norm_cfg. Bias of conv will be set as True if `norm_cfg` is
            None, otherwise False. Defaults to "auto".
        loss_cls (:obj:`ConfigDict` or dict): Config of classification loss.
        loss_bbox (:obj:`ConfigDict` or dict): Config of localization loss.
        loss_centerness (:obj:`ConfigDict`, or dict): Config of centerness
            loss.
        norm_cfg (:obj:`ConfigDict` or dict): dictionary to construct and
            config norm layer.  Defaults to
            ``norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)``.
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or \
            dict]): Initialization config dict.

    Example:
        >>> self = FCOSHead(11, 7)
        >>> feats = [torch.rand(1, 7, s, s) for s in [4, 8, 16, 32, 64]]
        >>> cls_score, bbox_pred, centerness = self.forward(feats)
        >>> assert len(cls_score) == len(self.scales)
    """  # noqa: E501
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.absolute = nn.Sigmoid()

        # Define dynamic MLP modules for classification and regression features
        self.dynamic_mlp_cls = FusionModule(inplanes=self.feat_channels, planes=self.feat_channels, hidden=32, num_layers=2, mlp_type='c')
        self.dynamic_mlp_reg = FusionModule(inplanes=self.feat_channels, planes=self.feat_channels, hidden=32, num_layers=2, mlp_type='c')

    def _init_layers(self) -> None:
        super()._init_layers()


    def forward_single(self, x: Tensor, scale: Scale, stride: int) -> Tuple[Tensor, Tensor, Tensor]:
        cls_feat = x
        reg_feat = x
        seg_feat = x

        # Process segmentation features
        for seg_layer in self.seg_convs:
            seg_feat = seg_layer(seg_feat)

        # Process classification features
        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)

        # Process regression features
        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_feat)

        # Reshape features for dynamic MLP
        N, C, H, W = seg_feat.shape
        seg_feat_reshape = seg_feat.view(N, C, -1).permute(0, 2, 1)

        cls_feat_reshape = cls_feat.view(N, C, -1).permute(0, 2, 1)
        reg_feat_reshape = reg_feat.view(N, C, -1).permute(0, 2, 1)

        # Apply dynamic MLP to classification and regression features
        cls_feat_reshape = self.dynamic_mlp_cls(cls_feat_reshape, seg_feat_reshape)
        reg_feat_reshape = self.dynamic_mlp_reg(reg_feat_reshape, seg_feat_reshape)

        # Reshape features back to original dimensions
        cls_feat = cls_feat_reshape.permute(0, 2, 1).view(N, C, H, W)
        reg_feat = reg_feat_reshape.permute(0, 2, 1).view(N, C, H, W)

        # Compute final outputs
        cls_score = self.conv_cls(cls_feat)
        bbox_pred = self.conv_reg(reg_feat)
        seg_score = self.conv_seg(seg_feat)

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

        seg_score = self.absolute(seg_score)

        return cls_score, bbox_pred, centerness, seg_score

class Basic1d(nn.Module):
    def __init__(self, inplanes, planes, with_activation=True):
        super().__init__()
        self.fc = nn.Linear(inplanes, planes)
        self.with_activation = with_activation
        if with_activation:
            self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.fc(x)
        if self.with_activation:
            x = self.act(x)
        return x

class Dynamic_MLP_A(nn.Module):
    def __init__(self, inplanes, planes, loc_planes):
        super().__init__()
        self.inplanes = inplanes
        self.planes = planes

        self.get_weight = nn.Linear(loc_planes, inplanes * planes)
        self.norm = nn.LayerNorm(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, img_fea, loc_fea):
        weight = self.get_weight(loc_fea)
        weight = weight.view(-1, self.inplanes, self.planes)

        img_fea = torch.bmm(img_fea.unsqueeze(1), weight).squeeze(1)
        img_fea = self.norm(img_fea)
        img_fea = self.relu(img_fea)

        return img_fea

class Dynamic_MLP_B(nn.Module):
    def __init__(self, inplanes, planes, loc_planes):
        super().__init__()
        self.inplanes = inplanes
        self.planes = planes

        self.conv11 = Basic1d(inplanes, inplanes, True)
        self.conv12 = nn.Linear(inplanes, inplanes)

        self.conv21 = Basic1d(loc_planes, inplanes, True)
        self.conv22 = nn.Linear(inplanes, inplanes * planes)

        self.br = nn.Sequential(
            nn.LayerNorm(planes),
            nn.ReLU(inplace=True),
        )
        self.conv3 = Basic1d(planes, planes, False)

    def forward(self, img_fea, loc_fea):
        weight11 = self.conv11(img_fea)
        weight12 = self.conv12(weight11)

        weight21 = self.conv21(loc_fea)
        weight22 = self.conv22(weight21).view(-1, self.inplanes, self.planes)

        img_fea = torch.bmm(weight12.unsqueeze(1), weight22).squeeze(1)
        img_fea = self.br(img_fea)
        img_fea = self.conv3(img_fea)

        return img_fea
   
class Dynamic_MLP_C(nn.Module):
    def __init__(self, inplanes, planes, loc_planes):
        super().__init__()
        self.inplanes = inplanes
        self.planes = planes

        self.conv11 = Basic1d(inplanes + loc_planes, inplanes, True)
        self.conv12 = nn.Linear(inplanes, inplanes)

        self.conv21 = Basic1d(inplanes + loc_planes, inplanes, True)
        self.conv22 = nn.Linear(inplanes, inplanes * planes)

        self.br = nn.Sequential(
            nn.LayerNorm(planes),
            nn.ReLU(inplace=True),
        )
        self.conv3 = Basic1d(planes, planes, False)

    def forward(self, img_fea, loc_fea):
        # Concatenate the image and location features
        B, N, C = img_fea.shape        
        
        cat_fea = torch.cat([img_fea, loc_fea], dim=2)

        # First set of convolutions
        weight11 = self.conv11(cat_fea)
        weight12 = self.conv12(weight11)

        # Second set of convolutions
        weight21 = self.conv21(cat_fea)
        weight22 = self.conv22(weight21).view(B, -1, self.inplanes, self.planes)

        # Apply dynamic weights
        img_fea = torch.matmul(weight12.unsqueeze(2), weight22).squeeze(2)  # Remove the added dimension
        img_fea = self.br(img_fea)
        img_fea = self.conv3(img_fea)

        return img_fea

class RecursiveBlock(nn.Module):
    def __init__(self, inplanes, planes, loc_planes, mlp_type='c'):
        super().__init__()
        if mlp_type.lower() == 'a':
            MLP = Dynamic_MLP_A
        elif mlp_type.lower() == 'b':
            MLP = Dynamic_MLP_B
        elif mlp_type.lower() == 'c':
            MLP = Dynamic_MLP_C

        self.dynamic_conv = MLP(inplanes, planes, loc_planes)

    def forward(self, img_fea, loc_fea):
        img_fea = self.dynamic_conv(img_fea, loc_fea)
        return img_fea, loc_fea

class FusionModule(nn.Module):
    def __init__(self, inplanes=256, planes=256, hidden=64, num_layers=1, mlp_type='c'):
        super().__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.hidden = hidden

        self.conv1 = nn.Linear(inplanes, planes)

        conv2 = []
        if num_layers == 1:
            conv2.append(RecursiveBlock(planes, planes, loc_planes=planes, mlp_type=mlp_type))
        else:
            conv2.append(RecursiveBlock(planes, hidden, loc_planesp=planes, mlp_type=mlp_type))
            for _ in range(1, num_layers - 1):
                conv2.append(RecursiveBlock(hidden, hidden, loc_planes=planes, mlp_type=mlp_type))
            conv2.append(RecursiveBlock(hidden, planes, loc_planes=planes, mlp_type=mlp_type))
        self.conv2 = nn.ModuleList(conv2)

        self.conv3 = nn.Linear(planes, inplanes)
        self.norm3 = nn.LayerNorm(inplanes)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, img_fea, loc_fea):
        identity = img_fea

        img_fea = self.conv1(img_fea)

        for m in self.conv2:
            img_fea, loc_fea = m(img_fea, loc_fea)

        img_fea = self.conv3(img_fea)
        img_fea = self.norm3(img_fea)

        img_fea += identity

        return img_fea
