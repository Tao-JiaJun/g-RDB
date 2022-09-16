"""
# Description: 计算focal loss工具类
# Author: Taojj
# Date: 2020-09-03 10:27:09
# LastEditTime: 2020-09-03 17:50:29
# FilePath: /FCOSLite/utils/focal_loss.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
class FocalWithLogitsLoss(nn.Module):
    def __init__(self, reduction='mean', gamma=2.0, alpha=0.25):
        super(FocalWithLogitsLoss, self).__init__()
        self.reduction = reduction
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(input=logits, 
                                                     target=targets, 
                                                     reduction="none"
                                                     )
        p_t = p * targets + (1.0 - p) * (1.0 - targets)
        loss = ce_loss * ((1.0 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            batch_size = logits.size(0)
            pos_inds = (targets == 1.0).float()
            # [B, H*W, C] -> [B,]
            num_pos = pos_inds.sum([1, 2]).clamp(1)
            loss = loss.sum([1, 2])
    
            loss = (loss / num_pos).sum() / batch_size

        elif self.reduction == "sum":
            loss = torch.sum(loss)

        return loss

class BCEFocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=0.6):
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha 

    def forward(self, inputs, targets):
        loss = self.alpha * (1.0-inputs)**self.gamma * (targets) * torch.log(inputs + 1e-14) + \
                (inputs)**self.gamma * (1.0 - targets) * torch.log(1.0 - inputs + 1e-14)
        loss = -torch.sum(torch.sum(loss, dim=-1), dim=-1)
        return loss


class HeatmapLoss(torch.nn.Module):
    def __init__(self, alpha=2, beta=4, reduction='mean'):
        super(HeatmapLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction
    def forward(self, inputs, targets):
        center_id = (targets == 1.0).float()
        other_id = (targets != 1.0).float()
        center_loss = -center_id * (1.0-inputs)**self.alpha * torch.log(inputs + 1e-14)
        other_loss = -other_id * (1 - targets)**self.beta * (inputs)**self.alpha * torch.log(1.0 - inputs + 1e-14)
        loss = center_loss + other_loss

        if self.reduction == 'mean':
            batch_size = loss.size(0)
            loss = torch.sum(loss) / batch_size

        if self.reduction == 'sum':
            loss = torch.sum(loss) / batch_size
        return loss
