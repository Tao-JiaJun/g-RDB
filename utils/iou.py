"""
# Description: 计算iou工具类
# Author: Tao
# Date: 2020-09-03 13:40:40
# LastEditTime: 2020-10-04 21:07:36
# FilePath: /FCOSLite/utils/iou.py
"""
import torch
import numpy as np
import math
import sys
sys.path.append("./")
#from utils.xyxyiou import CIoU
class IoU():
    def __init__(self, iou_type='iou'):
        assert iou_type in ['iou', 'giou', 'diou', 'ciou'], \
        'IoU mode should be either iou, giou, diou or ciou'
        self.iou_type = iou_type
    def __call__(self, pred, label, eps=1e-14):
        # 计算label的长宽
        w_label = (label[:, :, 0] + label[:, :, 2]).clamp(min=eps) # l+r
        h_label = (label[:, :, 1] + label[:, :, 3]).clamp(min=eps) # r+b
        # 计算pred的长宽
        w_pred = (pred[:, :, 0] + pred[:, :, 2]).clamp(min=eps) # l+r
        h_pred = (pred[:, :, 1] + pred[:, :, 3]).clamp(min=eps) # r+b
        # 计算label的面积
        s_label = w_label * h_label
        # 计算pred的面积
        s_pred = w_pred * h_pred
        # 计算相交的面积
        # left最短的边和right最短的边  相加则为相交部分的长
        w_inter = (torch.min(label[:, :, 0], pred[:, :, 0]) + torch.min(label[:, :, 2], pred[:, :, 2])).clamp(min=eps)
        # top最短的边和bottom最短的边相加则为相交部分的宽
        h_inter = (torch.min(label[:, :, 1], pred[:, :, 1]) + torch.min(label[:, :, 3], pred[:, :, 3])).clamp(min=eps)
        # 计算交集面积
        s_inter = w_inter * h_inter
        # 计算并集面积
        s_union = s_label + s_pred - s_inter + eps
        IoU = torch.true_divide(s_inter, s_union + eps)
        if self.iou_type == 'iou':
            return IoU
        # 计算最小凸包面积
        # left最长的边和right最长的边  相加则为最小凸包部分的长 enclosing
        w_enclose = torch.max(label[:, :, 0], pred[:, :, 0]) + torch.max(label[:, :, 2], pred[:, :, 2])
        # top最长的边和bottom最长的边  相加则为最小凸包部分的长
        h_enclose = torch.max(label[:, :, 1], pred[:, :, 1]) + torch.max(label[:, :, 3], pred[:, :, 3])
        s_enclose = w_enclose * h_enclose
        if self.iou_type == 'giou':
            GIoU = IoU - torch.true_divide(s_enclose - s_union, s_enclose + eps)
            return GIoU
        else:
            # 计算最小凸包对角线欧氏距离
            enclose_distance = torch.pow(w_enclose, 2) + torch.pow(h_enclose, 2)
            # 计算两个矩形中心点的欧氏距离
            # 两个框最小值
            min_dist = torch.min(label[:, :, :], pred[:, :, :])
            # 两个框最大值
            max_dist = torch.max(label[:, :, :], pred[:, :, :])
            # 分别计算(l+r)/2 (t+b)/2
            min_w = torch.true_divide(min_dist[:, :, 0] + min_dist[:, :, 2], 2) 
            min_h = torch.true_divide(min_dist[:, :, 1] + min_dist[:, :, 3], 2)
            max_w = torch.true_divide(max_dist[:, :, 0] + max_dist[:, :, 2], 2) 
            max_h = torch.true_divide(max_dist[:, :, 1] + max_dist[:, :, 3], 2)
            center_distance = torch.pow(max_w - min_w, 2) + torch.pow(max_h - min_h, 2)
            DIoU1 = IoU - torch.true_divide(center_distance, enclose_distance + eps)
            # 分别计算(l-r)/2 (t-b)/2
            min_w = torch.true_divide(min_dist[:, :, 0] - min_dist[:, :, 2], 2) 
            min_h = torch.true_divide(min_dist[:, :, 1] - min_dist[:, :, 3], 2)
            max_w = torch.true_divide(max_dist[:, :, 0] - max_dist[:, :, 2], 2) 
            max_h = torch.true_divide(max_dist[:, :, 1] - max_dist[:, :, 3], 2)
            center_distance = torch.pow(max_w - min_w, 2) + torch.pow(max_h - min_h, 2)
            DIoU2 = IoU - torch.true_divide(center_distance, enclose_distance + eps)
            # 利用0,1屏蔽不需要的值
            DIoU1 = DIoU1 * (s_enclose!=s_union) # 当两个矩形相交或完全不相交时
            DIoU2 = DIoU2 * (s_enclose==s_union) # 当一个矩形完全包含另一个矩形时
            DIoU = DIoU1 + DIoU2
            if self.iou_type == 'ciou':
                v = torch.true_divide(4.0 * torch.pow(torch.atan(w_label/ (h_label+ eps) ) - torch.atan(w_pred/ (h_pred + eps)), 2), (math.pi**2))
                a = torch.true_divide(v, (1.0 - IoU) + v + eps)
                CIoU = DIoU - a * v
                return CIoU
            else:
                return DIoU





if __name__=='__main__':
    _na = np.array([[1,1,5,5],[1,1,5,5],[1,1,5,5],[1,1,5,5],[1,1,5,5],[1,1,5,5]])
    _nb = np.array([[1,1,5,5],[0,0,1,1],[2,2,3,3],[0,4,2,5],[1,6,2,7],[4,3,7,6]])
    _a = torch.from_numpy(_na).float().view(6,4)
    _b = torch.from_numpy(_nb).float().view(6,4)

    na = np.array([[0,0,0,0],[0,0,4,4],[1,1,3,3],[1,3,3,1],[2,2,2,2],[3,3,1,1]])
    nb = np.array([[0,0,0,0],[1,1,0,0],[0,0,1,1],[2,0,0,1],[2,-3,-1,4],[0,1,3,2]])
    a = torch.from_numpy(na).float().view(1,6,4)
    b = torch.from_numpy(nb).float().view(1,6,4)

    # na = np.array([[1,1,3,3]])
    # nb = np.array([[0,0,1,1]])
    # a = torch.from_numpy(na).view(1,1,4)
    # b = torch.from_numpy(nb).view(1,1,4)
    iou = IoU(iou_type='ciou')
    #aa = IOULoss()
    # print("xyxy:")
    # print(giou(_a,_b))
    print("ltrb:")
    print(iou(a,b))
    # print("ltrb:")
    # print(giou_loss(a,b))


    # print(CIoU([1,1,5,5],[1,1,5,5]))
    # print(CIoU([1,1,5,5],[0,0,1,1]))
    # print(CIoU([1,1,5,5],[2,2,3,3]))
    # print(CIoU([1,1,5,5],[0,4,2,5]))
    # print(CIoU([1,1,5,5],[1,6,2,7]))
    # print(CIoU([1,1,5,5],[4,3,7,6]))