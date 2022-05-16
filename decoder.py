import torch
import torch.nn as nn
import torch.nn.functional as F

"""
original resolution is H × W

x0 : H × W
x1 : H/2 × W/2
x2 : H/4 × W/4
x3 : H/8 × W/8
x4 : H/16 × W/16

skip_channels = [channel of x1, channel of x2, channel of x3, channel of x4]
output_channels = [channel of y0, channel of y1, channel of y2, channel of y_3, channel of y4 (output)]
"""
class Decoder(nn.Module):
    def __init__(self, skip_channels, output_channels):
        super(Decoder,self).__init__()
        self.conv1 = nn.Conv2d(skip_channels[4]+output_channels[0],output_channels[1],kernel_size=3,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(output_channels[1])
        self.conv2 = nn.Conv2d(skip_channels[3]+output_channels[1],output_channels[2],kernel_size=3,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(output_channels[2])
        self.conv3 = nn.Conv2d(skip_channels[2]+output_channels[2],output_channels[3],kernel_size=3,padding=1,bias=False)
        self.bn3 = nn.BatchNorm2d(output_channels[3])
        self.conv4 = nn.Conv2d(skip_channels[1]+output_channels[3],output_channels[4],kernel_size=3,padding=1,bias=False)
        self.relu = nn.ReLU(True)

    def foward(self,x0,x1,x2,x3,x4):
        x = F.interpolate(size=x3.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, x3],dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = F.interpolate(size=x2.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, x2],dim=1)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = F.interpolate(size=x1.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, x1],dim=1)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = F.interpolate(size=x0.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, x0],dim=1)
        x = self.conv4(x)

        return x
