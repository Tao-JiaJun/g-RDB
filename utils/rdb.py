import torch
import torch.nn as nn
import sys
sys.path.append("./")
from utils.modules import ConvBnActivation, SeparableConvBnActivation, SeparableConv2d, channel_shuffle
from utils.cbam import ChannelAttention, SpatialAttention
from utils import get_parameter_number
from utils.repvgg_block import RepVGGBlock, RepVGGBlock1x1,SeparableRepVGGBlock

class gRDB(nn.Module):
    def __init__(self, planes, depth, groups, use_at=False, deploy=False):
        super(gRDB, self).__init__()
        assert depth >= 1
        modules = []
        self.groups = groups
        for i in range(depth):
            modules.append(RepVGGBlock(in_channels=planes, out_channels=planes, kernel_size=3,
                                stride=1, padding=1, groups=self.groups))  
        self.dense_layers = nn.Sequential(*modules)
        self.conv1x1 = RepVGGBlock1x1(planes * (depth+1), planes, kernel_size=1)
        if use_at:
            self.ca = ChannelAttention(planes)
            self.sp = SpatialAttention()
        self.relu = nn.ReLU(inplace=True)
        self.use_at = use_at
        self.depth = depth
     
    def forward(self, x):
        _out = channel_shuffle(self.dense_layers[0](x), self.groups)
        out = torch.cat([x, _out], dim=1)
        for i in range(1, self.depth):
            _out = self.dense_layers[i](_out)
            _out = channel_shuffle(self.relu(_out + x), self.groups)
            out = torch.cat([out, _out], dim=1)
        out = self.conv1x1(out)
        out = self.relu(out + x)
        if self.use_at:
            out = self.ca(out) * out
            out = self.relu(out + x)
            out = self.sp(out) * out
            out = self.relu(out + x)
        return out

if __name__=='__main__':
    device = torch.device("cpu")
    C3 = torch.randn(1, 128, 40, 40).to(device)
    neck = gRDB(planes=128,depth=3,groups=1,use_at=False,deploy=True)
    neck.to(device)
    fpn = neck(C3)
    
    for i in fpn:
        print(i.size())
    print(get_parameter_number(neck))
    print(neck)