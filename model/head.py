import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("./")
from utils import get_parameter_number
from utils.repvgg_block import RepVGGBlock, OnlyConvRepVGGBlock
from model.backbone.repvgg import repvgg_model_convert
import math
class Pred(nn.Module):
    def __init__(self,pred_planes, inplanes, outplanes, depth, is_cls=False, boost=False):
        super(Pred, self).__init__()
        modules = []
        modules.append(RepVGGBlock(in_channels=inplanes, out_channels=outplanes, kernel_size=3,
                    stride=1, padding=1, groups=1))
        for _ in range(depth-1):
            modules.append(RepVGGBlock(in_channels=outplanes, out_channels=outplanes, kernel_size=3,
                                       stride=1, padding=1, groups=1))
        modules.append(OnlyConvRepVGGBlock(outplanes, pred_planes, kernel_size=3, stride=1, padding=1,bias=True))
        self.layer = nn.Sequential(*modules)
        if is_cls:
            nn.init.constant_(self.layer[-1].rbr_dense.bias,-math.log((1 - 0.01) / 0.01))
            nn.init.constant_(self.layer[-1].rbr_1x1.bias,-math.log((1 - 0.01) / 0.01))
    def forward(self,x):
        return self.layer(x)
class Head(nn.Module):
    def __init__(self, num_cls, inplanes, outplanes, depth, boost=False):
        super(Head, self).__init__()
        self.boost = boost
     
        self.num_cls = num_cls
      
        self.totol_pred_num = num_cls + 4 + 1
  
        self.pred_cls_3 = Pred(num_cls, inplanes, outplanes, depth=depth, is_cls=True)
        self.pred_reg_3 = Pred(4 + 1,   inplanes, outplanes, depth=depth, is_cls=False)
        if not boost:
            self.pred_cls_4 = Pred(num_cls, inplanes, outplanes, depth=depth, is_cls=True)
            self.pred_reg_4 = Pred(4 + 1,   inplanes, outplanes, depth=depth, is_cls=False)

    def forward(self, features):
        if self.boost:
            F3 = features
            B = F3.shape[0]
            pred_cls_3 = self.pred_cls_3(F3).view(B, self.num_cls, -1)
            pred_reg_3 = self.pred_reg_3(F3).view(B, 4 + 1, -1)
            preds = torch.cat([pred_cls_3, pred_reg_3], dim=1).permute(0, 2, 1)
        else:
            F3, F4 = features[:]
            # get batch size
            B = F3.shape[0]
            pred_cls_3 = self.pred_cls_3(F3).view(B, self.num_cls, -1)
            pred_reg_3 = self.pred_reg_3(F3).view(B, 4 + 1, -1)
            pred_cls_4 = self.pred_cls_4(F4).view(B, self.num_cls, -1)
            pred_reg_4 = self.pred_reg_4(F4).view(B, 4 + 1, -1)
            # [B, C, WxH] --> [B, WxH, C]
            pred_3 = torch.cat([pred_cls_3, pred_reg_3], dim=1).permute(0, 2, 1)
            pred_4 = torch.cat([pred_cls_4, pred_reg_4], dim=1).permute(0, 2, 1)
            preds = torch.cat([pred_3,pred_4],dim=1)
        return preds
 

if __name__ == "__main__":
    device = torch.device("cpu")
    head = Head(20,192,128,depth=1,boost=True).to(device)
    #head = repvgg_model_convert(head)
    F3 = torch.randn(1, 192, 40, 40).to(device)
    F4 = torch.randn(1, 192, 20, 20).to(device)
    F5 = torch.randn(1, 192, 10, 10).to(device)
    F6 = torch.randn(1, 192, 5, 5).to(device)
    out = head(F3)
    for i in out:
        print(i.size())
    print(get_parameter_number(head))
