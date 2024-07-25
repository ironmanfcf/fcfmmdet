# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, Scale
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData
from torch import Tensor
from mmdet.structures import SampleList
from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures.bbox import bbox_overlaps
from mmdet.utils import (ConfigType, InstanceList, PixelList, MultiConfig, OptConfigType,
                         OptInstanceList, reduce_mean)
from ..task_modules.prior_generators import anchor_inside_flags
from ..task_modules.samplers import PseudoSampler
from ..utils import (filter_scores_and_topk, images_to_levels, multi_apply,
                     unmap)
from .anchor_head import AnchorHead
from .gfl_head import GFLHead
from mmdet.registry import MODELS, TASK_UTILS

@MODELS.register_module()
class GFLDeCoDetV16Head(GFLHead):
    """Generalized Focal Loss Head with Distributional Corners Detection (GFLDeCoDetV16).

    This class extends the GFLHead to include additional functionality.
    """

    def __init__(self,
                 seg_out_channels=1,
                 loss_seg: ConfigType = dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 mapping=False,
                 **kwargs) -> None:
        self.seg_out_channels = seg_out_channels
        super().__init__( **kwargs)
        self.loss_seg = MODELS.build(loss_seg)
        
        self.unfold = nn.Unfold(3, 1, (3 - 1) // 2, 1)
        
        
        self.mapping=mapping
        self.absolute = nn.Sigmoid()
        self.map = nn.ReLU()
        
        # 定义 involution 模块
        # self.inner_involution = involution(channels=self.feat_channels, kernel_size=7, stride=1)
       
        # 定义 involution 模块
        self.cross_involution_cls = cross_involution(channels=self.feat_channels, kernel_size=7, num_layers=1, stride=1)
        self.cross_involution_reg = cross_involution(channels=self.feat_channels, kernel_size=7, num_layers=1, stride=1)
        
    def _init_layers(self) -> None:
        """Initialize layers of the head."""
        super()._init_layers()
        self._init_seg_convs()
        self.conv_seg = nn.Conv2d(
            self.feat_channels, self.seg_out_channels, 3, padding=1)

    def _init_seg_convs(self) -> None:
        """Initialize classification conv layers of the head."""
        self.seg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.seg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
    
                        
    def loss(self, x: Tuple[Tensor], batch_data_samples: SampleList) -> dict:
        """Perform forward propagation and loss calculation of the detection
        head on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        outs = self(x)

        outputs = unpack_gt_instances_sem_seg(batch_data_samples)
        (batch_gt_instances, batch_gt_instances_ignore, batch_gt_sem_seg,
         batch_img_metas) = outputs

        loss_inputs = outs + (batch_gt_instances, batch_gt_sem_seg,
                              batch_img_metas, batch_gt_instances_ignore)
        losses = self.loss_by_feat(*loss_inputs)
        return losses
    
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
        return multi_apply(self.forward_single, x, self.scales)
    
    def forward_single(self, x: Tensor, scale: Scale) -> Sequence[Tensor]:
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.

        Returns:
            tuple:

            - cls_score (Tensor): Cls and quality joint scores for a single
              scale level the channel number is num_classes.
            - bbox_pred (Tensor): Box distribution logits for a single scale
              level, the channel number is 4*(n+1), n is max value of
              integral set.
        """
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
            
        # # 处理回归特征
        for i, reg_layer in enumerate(self.reg_convs):
            reg_feat_condi  = self.cross_involution_reg(reg_feat, seg_feats[i+1])  # 用seg_feats进行调制
            reg_feat = reg_feat_condi + reg_feat  # 恒等映射，可以在此处添加其他操作
            reg_feat = reg_layer(reg_feat)
            
        # 使用最后一层 seg_feat 进行分割映射
        seg_score = self.conv_seg(seg_feats[-1])
        if self.mapping:
            seg_score = self.map(seg_score)
        else: 
            seg_score = self.absolute(seg_score)
        
        # for cls_conv in self.cls_convs:
        #     cls_feat = cls_conv(cls_feat)
        # for reg_conv in self.reg_convs:
        #     reg_feat = reg_conv(reg_feat)
        cls_score = self.gfl_cls(cls_feat)
        bbox_pred = scale(self.gfl_reg(reg_feat)).float()
        return cls_score, bbox_pred, seg_score
    
    def loss_by_feat_single(self, anchors: Tensor, cls_score: Tensor,
                            bbox_pred: Tensor, labels: Tensor,
                            label_weights: Tensor, bbox_targets: Tensor,
                            stride: Tuple[int], avg_factor: int) -> dict:
        """Calculate the loss of a single scale level based on the features
        extracted by the detection head.

        Args:
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            cls_score (Tensor): Cls and quality joint scores for each scale
                level has shape (N, num_classes, H, W).
            bbox_pred (Tensor): Box distribution logits for each scale
                level with shape (N, 4*(n+1), H, W), n is max value of integral
                set.
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor with
                shape (N, num_total_anchors, 4).
            stride (Tuple[int]): Stride in this scale level.
            avg_factor (int): Average factor that is used to average
                the loss. When using sampling method, avg_factor is usually
                the sum of positive and negative priors. When using
                `PseudoSampler`, `avg_factor` is usually equal to the number
                of positive priors.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert stride[0] == stride[1], 'h stride is not equal to w stride!'
        anchors = anchors.reshape(-1, 4)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        bbox_pred = bbox_pred.permute(0, 2, 3,
                                      1).reshape(-1, 4 * (self.reg_max + 1))
        bbox_targets = bbox_targets.reshape(-1, 4)
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0)
                    & (labels < bg_class_ind)).nonzero().squeeze(1)
        score = label_weights.new_zeros(labels.shape)

        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]
            pos_anchors = anchors[pos_inds]
            pos_anchor_centers = self.anchor_center(pos_anchors) / stride[0]

            weight_targets = cls_score.detach().sigmoid()
            weight_targets = weight_targets.max(dim=1)[0][pos_inds]
            pos_bbox_pred_corners = self.integral(pos_bbox_pred)
            pos_decode_bbox_pred = self.bbox_coder.decode(
                pos_anchor_centers, pos_bbox_pred_corners)
            pos_decode_bbox_targets = pos_bbox_targets / stride[0]
            score[pos_inds] = bbox_overlaps(
                pos_decode_bbox_pred.detach(),
                pos_decode_bbox_targets,
                is_aligned=True)
            pred_corners = pos_bbox_pred.reshape(-1, self.reg_max + 1)
            target_corners = self.bbox_coder.encode(pos_anchor_centers,
                                                    pos_decode_bbox_targets,
                                                    self.reg_max).reshape(-1)

            # regression loss
            loss_bbox = self.loss_bbox(
                pos_decode_bbox_pred,
                pos_decode_bbox_targets,
                weight=weight_targets,
                avg_factor=1.0)

            # dfl loss
            loss_dfl = self.loss_dfl(
                pred_corners,
                target_corners,
                weight=weight_targets[:, None].expand(-1, 4).reshape(-1),
                avg_factor=4.0)
        else:
            loss_bbox = bbox_pred.sum() * 0
            loss_dfl = bbox_pred.sum() * 0
            weight_targets = bbox_pred.new_tensor(0)

        # cls (qfl) loss
        loss_cls = self.loss_cls(
            cls_score, (labels, score),
            weight=label_weights,
            avg_factor=avg_factor)

        return loss_cls, loss_bbox, loss_dfl, weight_targets.sum() 

    def loss_by_feat(
            self,
            cls_scores: List[Tensor],
            bbox_preds: List[Tensor],
            seg_scores: List[Tensor],
            batch_gt_instances: InstanceList,
            batch_gt_sem_seg: PixelList,
            batch_img_metas: List[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (list[Tensor]): Cls and quality scores for each scale
                level has shape (N, num_classes, H, W).
            bbox_preds (list[Tensor]): Box distribution logits for each scale
                level with shape (N, 4*(n+1), H, W), n is max value of integral
                set.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], Optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, batch_img_metas, device=device)

        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore)
        
############################################################
        seg_tagets = self.get_seg_targets(cls_scores, batch_gt_sem_seg)
        flatten_seg_scores = [
            seg_score.permute(0, 2, 3, 1).reshape(-1, self.seg_out_channels)
            for seg_score in seg_scores
        ]
        flatten_seg_scores = torch.cat(flatten_seg_scores)
        flatten_seg_targets = torch.cat(seg_tagets)
        
        loss_seg = self.loss_seg(
            flatten_seg_scores, flatten_seg_targets)
#################################################################

        (anchor_list, labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, avg_factor) = cls_reg_targets

        avg_factor = reduce_mean(
            torch.tensor(avg_factor, dtype=torch.float, device=device)).item()

        losses_cls, losses_bbox, losses_dfl,\
            avg_factor = multi_apply(
                self.loss_by_feat_single,
                anchor_list,
                cls_scores,
                bbox_preds,
                labels_list,
                label_weights_list,
                bbox_targets_list,
                self.prior_generator.strides,
                avg_factor=avg_factor)

        avg_factor = sum(avg_factor)
        avg_factor = reduce_mean(avg_factor).clamp_(min=1).item()
        losses_bbox = list(map(lambda x: x / avg_factor, losses_bbox))
        losses_dfl = list(map(lambda x: x / avg_factor, losses_dfl))
        return dict(
            loss_cls=losses_cls, loss_bbox=losses_bbox, loss_dfl=losses_dfl, loss_seg=loss_seg)
        
    def get_seg_targets(
        self,
        seg_scores: List[Tensor],
        batch_gt_sem_seg: InstanceList
    ) -> List[Tensor]:
        """
        Prepare segmentation targets.

        Args:
            seg_scores (List[Tensor]): List of segmentation scores of different
                levels in shape (batch_size, num_classes, h, w)
            batch_gt_instances (InstanceList): Ground truth instances.

        Returns:
            List[Tensor]: Segmentation targets of different levels.
        """
        # print("batch_gt_instances:", batch_gt_instances)
        # construct the segmentation targets of shape (B, C, H, W) for each layer
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
            lvl_seg_target = F.interpolate(lvl_seg_target,
                                           size=(h, w), mode='nearest')
            # lvl_seg_target[lvl_seg_target == 255] = 1.0
            seg_targets.append(lvl_seg_target)
        
        
        ###修改后的版本，解决由于数据增强以后导致的深度标签尺寸不一致问题。    
        # for lvl in range(lvls):
        #     _, _, h, w = seg_scores[lvl].shape
        #     lvl_seg_target = []
        #     for gt_sem_seg in batch_gt_sem_seg:
        #         seg_data = gt_sem_seg.sem_seg.data
        #         # Adjust to the correct size
        #         seg_data_resized = F.interpolate(seg_data.unsqueeze(0).float(), size=(h, w), mode='nearest')
        #         lvl_seg_target.append(seg_data_resized.squeeze(0))
        #     lvl_seg_target = torch.stack(lvl_seg_target, dim=0)
        #     seg_targets.append(lvl_seg_target)

        # flatten seg_targets
        flatten_seg_targets = [
            seg_target.permute(0, 2, 3, 1).reshape(-1, self.seg_out_channels)
            for seg_target in seg_targets
        ]

        return flatten_seg_targets
    
    
def unpack_gt_instances_sem_seg(batch_data_samples: SampleList) -> tuple:
    """Unpack ``gt_instances``, ``gt_instances_ignore``, ``img_metas``, and
    ``gt_sem_seg`` based on ``batch_data_samples``

    Args:
        batch_data_samples (List[:obj:`DetDataSample`]): The Data
            Samples. It usually includes information such as
            `gt_instance`, `gt_panoptic_seg`, `gt_sem_seg`, `gt_sem_seg`.

    Returns:
        tuple:

            - batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            - batch_gt_instances_ignore (list[:obj:`InstanceData`]):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
            - batch_gt_sem_seg (list[:obj:`PixelData`]): Batch of gt_sem_seg.
                It includes ``semantic_seg`` and ``ignore_index`` attributes.
            - batch_img_metas (list[dict]): Meta information of each image,
                e.g., image size, scaling factor, etc.
    """
    batch_gt_instances = []
    batch_gt_instances_ignore = []
    batch_gt_sem_seg = []
    batch_img_metas = []
    for data_sample in batch_data_samples:
        batch_img_metas.append(data_sample.metainfo)
        batch_gt_instances.append(data_sample.gt_instances)
        if 'ignored_instances' in data_sample:
            batch_gt_instances_ignore.append(data_sample.ignored_instances)
        else:
            batch_gt_instances_ignore.append(None)
        batch_gt_sem_seg.append(data_sample.gt_sem_seg)

    return (batch_gt_instances, batch_gt_instances_ignore, batch_gt_sem_seg,
            batch_img_metas)
    
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
    
class Integral(nn.Module):
    """A fixed layer for calculating integral result from distribution.

    This layer calculates the target location by :math: ``sum{P(y_i) * y_i}``,
    P(y_i) denotes the softmax vector that represents the discrete distribution
    y_i denotes the discrete set, usually {0, 1, 2, ..., reg_max}

    Args:
        reg_max (int): The maximal value of the discrete set. Defaults to 16.
            You may want to reset it according to your new dataset or related
            settings.
    """

    def __init__(self, reg_max: int = 16) -> None:
        super().__init__()
        self.reg_max = reg_max
        self.register_buffer('project',
                             torch.linspace(0, self.reg_max, self.reg_max + 1))

    def forward(self, x: Tensor) -> Tensor:
        """Forward feature from the regression head to get integral result of
        bounding box location.

        Args:
            x (Tensor): Features of the regression head, shape (N, 4*(n+1)),
                n is self.reg_max.

        Returns:
            x (Tensor): Integral result of box locations, i.e., distance
                offsets from the box center in four directions, shape (N, 4).
        """
        x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
        x = F.linear(x, self.project.type_as(x)).reshape(-1, 4)
        return x