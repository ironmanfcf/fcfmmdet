# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmdet.registry import MODELS
from .utils import weighted_loss


@weighted_loss
def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    """A Wrapper of MSE loss.
    Args:
        pred (Tensor): The prediction.
        target (Tensor): The learning target of the prediction.

    Returns:
        Tensor: loss Tensor
    """
    return F.mse_loss(pred, target, reduction='none')


@MODELS.register_module()
class MSELoss(nn.Module):
    """MSELoss.

    Args:
        reduction (str, optional): The method that reduces the loss to a
            scalar. Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of the loss. Defaults to 1.0
    """

    def __init__(self,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0,
                 ignore_index = None,
                 pad:bool=False) -> None:
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self.pad = pad

    def forward(self,
                pred: Tensor,
                target: Tensor,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None,
                ignore_index = -100) -> Tensor:
        """Forward function of loss.

        Args:
            pred (Tensor): The prediction.
            target (Tensor): The learning target of the prediction.
            weight (Tensor, optional): Weight of the loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.

        Returns:
            Tensor: The calculated loss.
        """
        
        if self.pad:
            # The default value of ignore_index is the same as in `F.cross_entropy`
            ignore_index = -100 if self.ignore_index is None else self.ignore_index
            
            # 打印处理之前的pred和target最大值和最小值
            # print("Before valid_index processing:")
            # print("pred max:", torch.max(pred).item(), "| pred min:", torch.min(pred).item())
            # print("target max:", torch.max(target).item(), "| target min:", torch.min(target).item())
            
            # Mask out ignored elements
            valid_index = ((target >= 0) & (target != ignore_index)).float()
            # Count the number of 0s and 1s in valid_mask
            # num_zeros = torch.sum(valid_index == 0).item()
            # num_ones = torch.sum(valid_index == 1).item()
                            
            # print("Number of 0s in valid_mask:", num_zeros)
            # print("Number of 1s in valid_mask:", num_ones)
                    
            pred = pred * valid_index
            target = target * valid_index
            # # 打印处理之后的 pred 和 target 最大值和最小值
            # print("After valid_index processing:")
            # print("pred max:", torch.max(pred).item(), "| pred min:", torch.min(pred).item())
            # print("target max:", torch.max(target).item(), "| target min:", torch.min(target).item())
        
        

        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss = self.loss_weight * mse_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss
