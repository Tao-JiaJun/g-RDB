import torch
import torch.nn as nn
import sys
sys.path.append("./")
from utils.repvgg_block import OnlyConvRepVGGBlock
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc1 = OnlyConvRepVGGBlock(in_planes, in_planes//16, 3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = OnlyConvRepVGGBlock(in_planes//16, in_planes, 3, padding=1)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention,self).__init__()
        self.conv1 = OnlyConvRepVGGBlock(2, 1, 3, padding=1)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


if __name__ == '__main__':
    x = torch.randn(1, 256, 40, 40)    # b, c, h, w
    ca_model = ChannelAttention(256)
    y = ca_model(x)
    print(y.shape)