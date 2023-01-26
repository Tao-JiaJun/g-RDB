import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
import os
import numpy as np
import math
import sys
sys.path.append("./")
from model import get_backbone, get_fpn
from model.head import Head
from model.loss import Loss
from model.ground_truth_maker import GTMaker
from utils.nms import nms
class Detector(nn.Module):
    def __init__(self, 
                 device,
                 input_size,
                 num_cls,
                 strides,
                 cfg,
                 boost=False):
        # 确定backbone neck 和 head
        super(Detector, self).__init__()
        self.backbone = get_backbone(name=cfg['BACKBONE']['NAME'],
                                     pretrained=cfg['BACKBONE']['PRETRAINED'])
        self.neck = get_fpn(param=cfg['NECK'],boost=boost)
        self.head = Head(num_cls=num_cls,
                         inplanes=cfg['NECK']['OUT_CHANNEL'],
                         outplanes=cfg['HEAD']['OUT_CHANNEL'],
                         depth=cfg['HEAD']['DEPTH'],
                         boost=boost)
        self.input_size = input_size
        if boost:
            self.strides = strides[0:1]
        else:
            self.strides = strides
        self.device = device
        self.conf_thresh = 0.05
        self.nms_thresh = 0.5
        self.scale = np.array([[self.input_size[1], self.input_size[0], self.input_size[1], self.input_size[0]]])
        self.topk = 100
        self.num_cls = num_cls
        self.use_nms = True
        self.location_weight = torch.tensor([[-1, -1, 1, 1]]).float().to(self.device)
        self.pixel_location = self.set_init().to(self.device)
        self.gt_maker = GTMaker(input_size, num_cls, strides)
        self.loss = Loss()
        self.conf_tensor = torch.ones_like(torch.empty(self.topk).to(self.device)) *self.conf_thresh
        self.relu = torch.nn.ReLU()
        self.nks_alpha = 0.6

  
    def forward(self, x, gt_list=None):
        # backbone
        tic = time.time()
        features = self.backbone(x)
        features = self.neck(features)
        features = self.head(features)
        toc = time.time() - tic
        if self.training:
            gt_tensor = torch.from_numpy(self.gt_maker(gt_list=gt_list)).float().to(self.device)
            return self.loss(features, gt_tensor, num_cls=self.num_cls)
        else:
            return toc, self.evaluating(features)
    
    def evaluating(self, pred_head):
        """
        # Args: 
        #   pred_head
        # Returns: 
        #   cls    
        #   bboxes 
        #   scores 
        """
        with torch.no_grad():
            # batch size
            bbox_list = []
            score_list = []
            cls_list = []
            B = pred_head.shape[0]
            cls_pred = pred_head[:, :, : self.num_cls].sigmoid()
            # [xmin, ymin, xmax, ymax] = [x,y,x,y] + [l,t,r,b] * [-1,-1,1,1]
            loc_pred = self.relu(pred_head[:, :, self.num_cls:-1]) * self.location_weight + self.pixel_location
            nks_pred = pred_head[:, :, -1].sigmoid().unsqueeze(-1)
            nks_pred = torch.true_divide(1,1+torch.exp(-2*(nks_pred)+1))
            cls_pred = torch.pow(cls_pred, (2-nks_pred)*self.nks_alpha+1e-14)
           
            for i in range(B):
                # simple nms
                hmax = F.max_pool2d(cls_pred, kernel_size=3, padding=1, stride=1)
                keep = (hmax == cls_pred).float()
                cls_pred *= keep
                topk_scores, topk_inds, topk_clses = self._topk(cls_pred[i, ...])
                keep = torch.gt(topk_scores,self.conf_tensor)
                topk_scores = topk_scores[keep].cpu().numpy()
                topk_cls = topk_clses[keep].cpu().numpy()
                topk_bbox_pred =  loc_pred[i, ...][topk_inds[keep]].cpu().numpy()

                if self.use_nms:
                    # nms
                    keep = np.zeros(len(topk_bbox_pred), dtype=np.int)
                    for i in range(self.num_cls):
                        inds = np.where(topk_cls == i)[0]
                        if len(inds) == 0:
                            continue
                        c_bboxes = topk_bbox_pred[inds]
                        c_scores = topk_scores[inds]
                        c_keep = nms(c_bboxes, c_scores,self.nms_thresh)
                        keep[inds[c_keep]] = 1

                    keep = np.where(keep > 0)
                    topk_bbox_pred = topk_bbox_pred[keep]
                    topk_scores = topk_scores[keep]
                    topk_cls = topk_cls[keep]
                
                bboxes = np.clip(topk_bbox_pred / self.scale, 0, 1)
                bbox_list.append(bboxes)
                score_list.append(topk_scores)
                cls_list.append(topk_cls)
            return bbox_list, score_list, cls_list
    def postprocess(self, bboxes, scores):
        """
        bboxes: (HxW, 4), bsize = 1
        scores: (HxW, num_classes), bsize = 1
        """

        cls_inds = np.argmax(scores, axis=1)
        scores = scores[(np.arange(scores.shape[0]), cls_inds)]
        
        # threshold
        keep = np.where(scores >= self.conf_thresh)
        bboxes = bboxes[keep]
        scores = scores[keep]
        cls_inds = cls_inds[keep]

        # NMS
        keep = np.zeros(len(bboxes), dtype=np.int)
        for i in range(self.num_cls):
            inds = np.where(cls_inds == i)[0]
            if len(inds) == 0:
                continue
            c_bboxes = bboxes[inds]
            c_scores = scores[inds]
            c_keep = self.nms(c_bboxes, c_scores)
            keep[inds[c_keep]] = 1

        keep = np.where(keep > 0)
        bboxes = bboxes[keep]
        scores = scores[keep]
        cls_inds = cls_inds[keep]

        return bboxes, scores, cls_inds


    def _topk(self, scores):
        HW, C  = scores.size()
        scores = scores.permute(1, 0).contiguous().view(-1)
     
        topk_scores, topk_inds = torch.topk(scores, self.topk)

        topk_clses = torch.true_divide(topk_inds, HW).int()
     
        topk_inds = topk_inds % (HW)
        return topk_scores, topk_inds, topk_clses
    def set_init(self):
        # 计算H*W个数
        total = sum([(self.input_size[0] // s) * (self.input_size[1] // s)
                     for s in self.strides])
        # 全部置为0
        pixel_location = torch.zeros(total, 4).to(self.device)
        start_index = 0
        for index in range(len(self.strides)):
            s = self.strides[index]
            # make a feature map size corresponding the scale
            ws = self.input_size[1] // s
            hs = self.input_size[0] // s
            for ys in range(hs):
                for xs in range(ws):
                    x_y = ys * ws + xs
                    index = x_y + start_index
                    x = xs * s + s / 2
                    y = ys * s + s / 2
                    pixel_location[index, :] = torch.tensor([[x, y, x, y]]).float()
            start_index += ws * hs
        return pixel_location

    def clip_boxes(self, boxes, im_shape):
        """
        Clip boxes to image boundaries.
        """
        if boxes.shape[0] == 0:
            return boxes

        # x1 >= 0
        boxes[:, 0::4] = np.maximum(np.minimum(
            boxes[:, 0::4], im_shape[1] - 1), 0)
        # y1 >= 0
        boxes[:, 1::4] = np.maximum(np.minimum(
            boxes[:, 1::4], im_shape[0] - 1), 0)
        # x2 < im_shape[1]
        boxes[:, 2::4] = np.maximum(np.minimum(
            boxes[:, 2::4], im_shape[1] - 1), 0)
        # y2 < im_shape[0]
        boxes[:, 3::4] = np.maximum(np.minimum(
            boxes[:, 3::4], im_shape[0] - 1), 0)
        return boxes

    def nms(self, dets, scores):
        """"Pure Python NMS baseline."""
        x1 = dets[:, 0]  # xmin
        y1 = dets[:, 1]  # ymin
        x2 = dets[:, 2]  # xmax
        y2 = dets[:, 3]  # ymax

        areas = (x2 - x1) * (y2 - y1)                 # the size of bbox
        # sort bounding boxes by decreasing order
        order = scores.argsort()[::-1]

        # store the final bounding boxes
        keep = []
        while order.size > 0:
            i = order[0]  # the index of the bbox with highest confidence
            keep.append(i)  # save it to keep
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(1e-28, xx2 - xx1)
            h = np.maximum(1e-28, yy2 - yy1)
            inter = w * h

            # Cross Area / (bbox + particular area - Cross Area)
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            # reserve all the boundingbox whose ovr less than thresh
            inds = np.where(ovr <= self.nms_thresh)[0]
            order = order[inds + 1]

        return keep