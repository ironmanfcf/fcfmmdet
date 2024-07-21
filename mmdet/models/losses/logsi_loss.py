    # def define_cost(self, pred, y0, m0):
    #     bsize = self.bsize
    #     npix = int(np.prod(test_shape(y0)[1:]))
    #     y0_target = y0.reshape((self.bsize, npix))
    #     y0_mask = m0.reshape((self.bsize, npix))
    #     pred = pred.reshape((self.bsize, npix))

    #     p = pred * y0_mask
    #     t = y0_target * y0_mask

    #     d = (p - t)

    #     nvalid_pix = T.sum(y0_mask, axis=1)
    #     depth_cost = (T.sum(nvalid_pix * T.sum(d**2, axis=1))
    #                      - 0.5*T.sum(T.sum(d, axis=1)**2)) \
    #                  / T.maximum(T.sum(nvalid_pix**2), 1)

    #     return depth_cost
    
    
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional
from mmdet.registry import MODELS



@MODELS.register_module()
class SILOGLOSS(nn.Module):
    """Scale-Invariant Logarithmic (SiLog) Loss.

    Args:
        bsize (int): Batch size.
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of the loss.
        avg_non_ignore (bool, optional): If True, only consider non-ignored elements for averaging. Defaults to False.
    """

    def __init__(self,
                 ignore_index = 255,
                 loss_weight: float = 1.0,
                 smooth: bool = False,
                 epsilon:  float=0.1,
                 log:bool=False) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.loss_weight = loss_weight
        self.smooth = smooth
        self.epsilon = epsilon
        self.log = log

    def forward(self,
                pred: Tensor,
                label: Tensor,
                ignore_index: int = 255
                        ) -> Tensor:
        """Forward function.

        Args:
            pred (Tensor): The prediction.
            label (Tensor): The target tensor.
            mask (Tensor): The mask tensor.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.

        Returns:
            Tensor: Calculated loss.
        """
        
        # The default value of ignore_index is the same as in `F.cross_entropy`
        ignore_index = -100 if ignore_index is None else ignore_index

        # Mask out ignored elements
        valid_mask = ((label >= 0) & (label != ignore_index)).float()
        
        pred_valid = pred * valid_mask
        label_valid = label * valid_mask
        if self.log:
            # Ensure pred_valid and label_valid are positive before taking log
            pred_valid = torch.clamp(pred_valid, min=1e-6)
            label_valid = torch.clamp(label_valid, min=1e-6)
            pred_valid = torch.log(pred_valid)
            label_valid = torch.log(label_valid)
        if self.smooth:
            label_valid = (1 - self.epsilon) * label_valid + self.epsilon * pred_valid

        diff = (pred_valid - label_valid)
        
        nvalid_pix=torch.sum(valid_mask)

        depth_cost = (torch.sum(nvalid_pix * torch.sum(diff**2))
                      - 0.5 * torch.sum(torch.sum(diff)**2)) \
                     / torch.maximum(torch.sum(nvalid_pix**2), torch.tensor(1.0, device=label.device))


        return self.loss_weight * depth_cost
