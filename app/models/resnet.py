import torch.nn as nn
from torchvision.models.resnet import ResNet, Bottleneck

class ResNet50(ResNet):
    def __init__(self, in_channels, norm_layer=None):
        super().__init__(
            block=Bottleneck,
            layers=[3, 4, 6, 3],
            replace_stride_with_dilation=[False, False, True],
            norm_layer=norm_layer
        )

        if in_channels != 3:
            self.conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)

        del self.avgpool
        del self.fc

    def forward(self, x):
        x0 = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x1 = x
        x = self.maxpool(x)
        x = self.layer1(x)
        x2 = x
        x = self.layer2(x)
        x3 = x
        x = self.layer3(x)
        x = self.layer4(x)
        x4 = x
        return x0, x1, x2, x3, x4