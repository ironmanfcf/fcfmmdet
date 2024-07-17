# Copyright (c) GrokCV. All rights reserved.
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmcv.cnn import ConvModule, Scale
from mmdet.registry import MODELS
from .fcos_seg_head import FCOSDepthHead
from mmdet.utils import (ConfigType, InstanceList, PixelList, MultiConfig,
                         OptInstanceList, RangeType, reduce_mean)
from mmcv.cnn import Conv2d, ConvModule
from mmdet.models.utils import multi_apply

@MODELS.register_module()
class FCOSDeCoDetHeadV18(FCOSDepthHead):
    def __init__(self, num_layers=1, **kwargs) -> None:
        super().__init__(**kwargs)
        self.feature_interaction_layers = num_layers
        self.feature_interaction_layers = num_layers
        self.absolute = nn.Sigmoid()
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
        

    def _init_layers(self) -> None:
        super()._init_layers()
        
    def forward(
            self, x: Tuple[Tensor]
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor]]:
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: A tuple of each level outputs.

            - cls_scores (list[Tensor]): Box scores for each scale level, \
            each is a 4D-tensor, the channel number is \
            num_points * num_classes.
            - bbox_preds (list[Tensor]): Box energies / deltas for each \
            scale level, each is a 4D-tensor, the channel number is \
            num_points * 4.
            - centernesses (list[Tensor]): centerness for each scale level, \
            each is a 4D-tensor, the channel number is num_points * 1.
            - seg_logits (list[Tensor]): segmentation logits for each scale level, \
            each is a 4D-tensor, the channel number is num_points * 1.
        """
        return multi_apply(self.forward_single, x, self.scales, self.strides)
 
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

        seg_score = self.absolute(seg_score)

        return cls_score, bbox_pred, centerness, seg_score
    
    def loss_by_feat(
        self,
        cls_scores: List[Tensor],
        bbox_preds: List[Tensor],
        centernesses: List[Tensor],
        seg_scores: List[Tensor],
        batch_gt_instances: InstanceList,
        batch_gt_sem_seg: PixelList,
        batch_img_metas: List[dict],
        batch_gt_instances_ignore: OptInstanceList = None,
    ) -> Dict[str, Tensor]:
        """
        Calculate the loss based on the features extracted by the detection head.
        """
        assert len(cls_scores) == len(bbox_preds) == len(centernesses) == len(seg_scores)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=bbox_preds[0].dtype,
            device=bbox_preds[0].device)
        labels, bbox_targets = self.get_targets(all_level_points, batch_gt_instances)

        num_imgs = cls_scores[0].size(0)
        flatten_cls_scores = [cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels) for cls_score in cls_scores]
        flatten_bbox_preds = [bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4) for bbox_pred in bbox_preds]
        flatten_centerness = [centerness.permute(0, 2, 3, 1).reshape(-1) for centerness in centernesses]

        # Upsample seg_scores to the size of ground truth targets
        gt_size = batch_gt_sem_seg[0].sem_seg.data.shape[-2:]
        upsampled_seg_scores = [F.interpolate(seg_score, size=gt_size, mode='bilinear', align_corners=False) for seg_score in seg_scores]

        flatten_seg_scores = [seg_score.permute(0, 2, 3, 1).reshape(-1, self.seg_out_channels) for seg_score in upsampled_seg_scores]

        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_centerness = torch.cat(flatten_centerness)
        flatten_seg_scores = torch.cat(flatten_seg_scores)

        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        
        seg_targets = self.get_seg_targets(seg_scores, batch_gt_sem_seg)
        flatten_seg_targets = torch.cat(seg_targets)

        flatten_points = torch.cat([points.repeat(num_imgs, 1) for points in all_level_points])

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels >= 0) & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        num_pos = max(reduce_mean(torch.tensor(len(pos_inds), dtype=torch.float, device=bbox_preds[0].device)), 1.0)

        loss_cls = self.loss_cls(flatten_cls_scores, flatten_labels, avg_factor=num_pos)
        loss_seg = self.loss_seg(flatten_seg_scores, flatten_seg_targets)

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]
        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        pos_centerness_targets = self.centerness_target(pos_bbox_targets)
        centerness_denorm = max(reduce_mean(pos_centerness_targets.sum().detach()), 1e-6)

        if len(pos_inds) > 0:
            pos_points = flatten_points[pos_inds]
            pos_decoded_bbox_preds = self.bbox_coder.decode(pos_points, pos_bbox_preds)
            pos_decoded_target_preds = self.bbox_coder.decode(pos_points, pos_bbox_targets)
            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds,
                weight=pos_centerness_targets,
                avg_factor=centerness_denorm)
            loss_centerness = self.loss_centerness(pos_centerness, pos_centerness_targets, avg_factor=num_pos)
        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_centerness = pos_centerness.sum()

        return dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_centerness=loss_centerness,
            loss_seg=loss_seg)

    def get_seg_targets(self, seg_scores: List[Tensor], batch_gt_sem_seg: InstanceList) -> List[Tensor]:
        """
        Prepare segmentation targets without downsampling.

        Args:
            seg_scores (List[Tensor]): List of segmentation scores of different levels in shape (batch_size, num_classes, h, w)
            batch_gt_instances (InstanceList): Ground truth instances.

        Returns:
            List[Tensor]: Segmentation targets of different levels.
        """
        lvls = len(seg_scores)
        batch_size = len(batch_gt_sem_seg)
        assert batch_size == seg_scores[0].size(0)
        seg_targets = []
            
        for lvl in range(lvls):
            _, _, h, w = seg_scores[lvl].shape
            lvl_seg_target = []
            for gt_sem_seg in batch_gt_sem_seg:
                lvl_seg_target.append(gt_sem_seg.sem_seg.data)
            lvl_seg_target = torch.stack(lvl_seg_target, dim=0)
                # lvl_seg_target = F.interpolate(lvl_seg_target,
                #                             size=(h, w), mode='nearest')
            seg_targets.append(lvl_seg_target)

        flatten_seg_targets = [
            seg_target.permute(0, 2, 3, 1).reshape(-1, self.seg_out_channels)
            for seg_target in seg_targets
        ]
        return flatten_seg_targets    

    

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