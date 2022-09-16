import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
sys.path.append("./")
from utils import get_parameter_number
from utils.rdb import gRDB
from utils.repvgg_block import RepVGGBlock, RepVGGBlock1x1
from model.backbone.repvgg import repvgg_model_convert
class RDBFPN(nn.Module):
    def __init__(self, inplanes:list, feature_size, depth=3, groups=4, use_at=False, boost=False):
        super().__init__()
        C3_size, C4_size = inplanes[:]
        self.conv3x3_6 = RepVGGBlock(in_channels=feature_size//2, out_channels=feature_size//2, kernel_size=3,stride=2, padding=1, groups=1)
        self.conv3x3_5 = RepVGGBlock(in_channels=C4_size, out_channels=feature_size//2, kernel_size=3,stride=2, padding=1, groups=1)
        self.conv1x1_4 = RepVGGBlock1x1(C4_size, feature_size//2, 1)
        self.conv1x1_3 = RepVGGBlock1x1(C3_size, feature_size//2, 1)
        if C3_size == feature_size//2:
            self.use_conv1x1_3 = False
        else:
            self.use_conv1x1_3 = True
        self.up_5 = RepVGGBlock1x1(feature_size, feature_size//2, 1) 
        self.up_4 = RepVGGBlock1x1(feature_size, feature_size//2, 1) 

        self.rdb_6 = gRDB(planes=feature_size//2, depth=depth, groups=groups, use_at=False)
        self.rdb_5 = gRDB(planes=feature_size,    depth=depth, groups=groups, use_at=False)
        self.rdb_4 = gRDB(planes=feature_size,    depth=depth, groups=groups, use_at=False)
        self.rdb_3 = gRDB(planes=feature_size,    depth=depth, groups=groups, use_at=use_at)
        
        self.relu = nn.ReLU(inplace=True)
        self.boost = boost
        print("use attetion:", use_at)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, features):

        C3, C4 = features[:]
        
        C5 = self.conv3x3_5(C4)
        C6 = self.conv3x3_6(C5)

        D6 = self.rdb_6(C6)
        D6_up = F.interpolate(D6, scale_factor=2.0, mode='bilinear', align_corners=True) 
        
        C5 = torch.cat([C5, D6_up], dim=1) #torch.cat([C5, D6_up], dim=1) 
        D5 = self.rdb_5(C5)
        D5_up = F.interpolate(self.up_5(D5), scale_factor=2.0, mode='bilinear', align_corners=True) 

        C4 = self.conv1x1_4(C4)
        C4 = torch.cat([C4, D5_up], dim=1) ##torch.cat([C4, D5_up], dim=1) 
        D4 = self.rdb_4(C4)
        D4_up = F.interpolate(self.up_4(D4), scale_factor=2.0, mode='bilinear', align_corners=True) 

        if self.use_conv1x1_3:
            C3 = self.conv1x1_3(C3)
        C3 = torch.cat([C3, D4_up], dim=1) 
        D3 = self.rdb_3(C3) 

        if self.boost:
            return D3
        else: 
            return D3, D4
from thop import profile
from thop import clever_format

if __name__=='__main__':

   
    device = "cpu"

    C3 = torch.randn(1, 96, 40, 40).to(device)
    C4 = torch.randn(1, 192, 20, 20).to(device)
    
    neck = RDBFPN(inplanes=[96,192],feature_size=128,depth=3,groups=2,use_at=False)
    print(get_parameter_number(neck))
    neck = repvgg_model_convert(neck)
    neck.to(device)
    macs, params = profile(neck, inputs=[C3,C4] )
    print(macs,params)
    fpn = neck([C3, C4])
    for i in fpn:
        print(i.size())
    print(get_parameter_number(neck))

    #print(neck)