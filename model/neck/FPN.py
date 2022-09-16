import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("./")
from utils import get_parameter_number
class FPN(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size):
        super(FPN, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        
        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)

        # # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        # self.P7_1 = nn.ReLU()
        # self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):
        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = F.interpolate(P5_x, scale_factor=2.0, mode='bilinear', align_corners=True) 
        P5_x = self.P5_2(P5_x)
     
            
        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = F.interpolate(P4_x, scale_factor=2.0, mode='bilinear', align_corners=True) 
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        P6_x = self.P6(C5)

        # P7_x = self.P7_1(P6_x)
        # P7_x = self.P7_2(P7_x)

        return [P3_x, P4_x, P5_x, P6_x]


if __name__=='__main__':
    device = torch.device("cpu")
    C3 = torch.randn(1, 128, 40, 40).to(device)
    C4 = torch.randn(1, 256, 20, 20).to(device)
    C5 = torch.randn(1, 512, 10, 10).to(device)
    neck = FPN(128,256,512,128)
    neck.to(device)
    fpn = neck([C3, C4, C5])
    for i in fpn:
        print(i.size())
    print(get_parameter_number(neck))