import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional
from mmdet.registry import MODELS

def knowledge_distillation_kl_div_loss(pred: Tensor,
                                       soft_label: Tensor,
                                       valid_mask,
                                       T: int,
                                       detach_target: bool = True) -> Tensor:
    r"""Loss function for knowledge distilling using KL divergence.

    Args:
        pred (Tensor): Predicted logits with shape (N, n + 1).
        soft_label (Tensor): Target logits with shape (N, N + 1).
        T (int): Temperature for distillation.
        detach_target (bool): Remove soft_label from automatic differentiation

    Returns:
        Tensor: Loss tensor with shape (N,).
    """
    assert pred.size() == soft_label.size()
    # target = F.softmax(soft_label / T, dim=1)
    target = soft_label
    if detach_target:
        target = target.detach()

    kd_loss = F.kl_div(
        # F.log_softmax(pred / T, dim=1),
        torch.sigmoid(pred),
        target, reduction='none')
    kd_loss = torch.abs(kd_loss)
    
    nvalid_pix = torch.sum(valid_mask)
    kd_loss=torch.sum(kd_loss * valid_mask)/torch.maximum(nvalid_pix, torch.tensor(1.0, device=soft_label.device)) * (T * T)
    

    return kd_loss

@MODELS.register_module()
class BiKLDLOSS(nn.Module):
    """Scale-Invariant Logarithmic (SiLog) Loss with Knowledge Distillation KL Divergence.

    Args:
        ignore_index (int, optional): Index that will be ignored in the loss computation. Defaults to 255.
        loss_weight (float, optional): The weight of the loss. Defaults to 1.0.
        smooth (bool, optional): If True, apply label smoothing. Defaults to False.
        epsilon (float, optional): Smoothing parameter. Defaults to 0.1.
        log (bool, optional): If True, apply logarithmic transformation to predictions and labels. Defaults to False.
        bi (bool, optional): If True, apply bidirectional KL divergence. Defaults to False.
        T (int, optional): Temperature for distillation. Defaults to 1.
    """

    def __init__(self,
                 ignore_index: int = 255,
                 loss_weight: float = 1.0,
                 smooth: bool = False,
                 epsilon: float = 0.1,
                 log: bool = False,
                 bi: bool = False,
                 T: int = 1) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.loss_weight = loss_weight
        self.smooth = smooth
        self.epsilon = epsilon
        self.log = log
        self.bi = bi
        self.T = T

    def forward(self,
                pred: Tensor,
                label: Tensor,
                soft_label: Optional[Tensor] = None,
                ignore_index: int = 255) -> Tensor:
        """Forward function.

        Args:
            pred (Tensor): The prediction.
            label (Tensor): The target tensor.
            soft_label (Tensor, optional): The soft target tensor for knowledge distillation.
            ignore_index (int, optional): Index that will be ignored in the loss computation. Defaults to 255.

        Returns:
            Tensor: Calculated loss.
        """

        # The default value of ignore_index is the same as in `F.cross_entropy`
        ignore_index = -100 if ignore_index is None else ignore_index

        # Mask out ignored elements
        valid_mask = ((label >= 0) & (label != ignore_index)).float()
        
        if self.log:
            # Ensure pred and label are positive before taking log
            pred = torch.clamp(pred, min=1e-6)
            label = torch.clamp(label, min=1e-6)
            pred = torch.log(pred)
            label = torch.log(label)
        if self.smooth:
            label = (1 - self.epsilon) * label + self.epsilon * pred

        # if self.bi:
        #     kl_div = F.kl_div(torch.clamp(pred, min=1e-6), 
        #                       torch.clamp(label, min=1e-6), 
        #                       reduction='none', log_target=False)
        #     kl_div_sym = F.kl_div(torch.clamp(1 - pred, min=1e-6), 
        #                           torch.clamp(1 - label, min=1e-6), 
        #                           reduction='none', log_target=False)
        #     total_divergence = torch.sum((torch.abs(kl_div) + torch.abs(kl_div_sym)) * valid_mask)
        # else:
        #     kl_div = F.kl_div(torch.clamp(pred, min=1e-6), 
        #                       torch.clamp(label, min=1e-6), 
        #                       reduction='none', log_target=False)
        #     total_divergence = torch.sum(torch.abs(kl_div) * valid_mask)
        
        # nvalid_pix = torch.sum(valid_mask)
        # depth_cost = total_divergence / torch.maximum(nvalid_pix, torch.tensor(1.0, device=label.device))

        # Knowledge Distillation Loss
        kd_loss = knowledge_distillation_kl_div_loss(pred, label,valid_mask, T=self.T)
        depth_cost = kd_loss

        return self.loss_weight * depth_cost