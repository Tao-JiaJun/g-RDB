import torch
from utils.focal_loss import BCEFocalLoss,HeatmapLoss,FocalWithLogitsLoss
from utils.iou import IoU
import torch.nn.functional as F
class Loss(object):
    def __init__(self, iou_type='iou'):
        self.cls_loss_func = HeatmapLoss(reduction='none')
        self.BCELoss = torch.nn.BCELoss(reduction='none')
        self.compute_iou = IoU(iou_type=iou_type)
        self.relu = torch.nn.ReLU()
        self.eps = 1e-14
    def __call__(self, pred, gt, num_cls):
        """
        # Args: 
        #   pred    : [num_cls + ltrb + nks + iou]
        #   gt      : [num_cls + ltrb + nks]
        #   num_cls : 20/80
        # Returns: 
        """

        pred_cls = pred[:, :, : num_cls].sigmoid()
        pred_reg = self.relu(pred[:, :, num_cls:-1])
        pred_nks = pred[:, :, -1].sigmoid()

        gt_cls = gt[:, :, : num_cls].float()
        gt_reg = gt[:, :, num_cls:-1].float()
        gt_nks = gt[:, :, -1].float()

        gt_pos = (gt_nks == 1.0).float()
        num_pos = gt_pos.sum(-1, keepdim=True).clamp(1)
        batch_size = pred_cls.size(0)
        
        cls_loss = torch.mean(torch.sum(torch.sum(self.cls_loss_func(pred_cls, gt_cls), dim=-1), dim=-1)  / num_pos)
        
        iou = self.compute_iou(pred_reg, gt_reg)
        reg_loss = ((1. - iou+ self.eps) * gt_pos / num_pos).sum() / batch_size

        gt_pos = (gt_nks >= 0.).float()
        num_pos = gt_pos.sum(-1, keepdim=True).clamp(1)
        nks_loss = (self.BCELoss(pred_nks, gt_nks) * gt_pos / num_pos).sum() / batch_size
        return cls_loss, reg_loss, nks_loss
    