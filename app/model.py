from turtle import forward
import torch
import torch.nn as nn
from torchvision.models.segmentation.deeplabv3 import ASPP
from resnet import ResNet50
from decoder import Decoder
from refine import Refine


class Base(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        self.backbone = ResNet50(input_channels)
        self.aspp = ASPP(2048, [3, 6, 9])
        self.decoder = Decoder([input_channels, 64, 256, 512], [256, 128, 64, 48, output_channels])

    def forward(self, x):
        x0, x1, x2, x3, x4 = self.backbone(x)
        x4 = self.aspp(x4)
        x = self.decoder(x0, x1, x2, x3, x4)
        return x

    def load_deeplabv3_pretrained_state_dict(self, state_dict):
        """
        Load pretrained DeepLabV3 state_dict and convert to match with our model structure.
        """
        #convert some state_dict item to aspp
        state_dict = {key.replace('classifier.classifier.0', 'aspp'): val for key, val in state_dict.items()}

        #only load weights that matched in key and shape. Ignore other weights.
        matched , total = 0
        original_state_dict = self.state_dict()
        for key in original_state_dict.keys():
            total +=1
            if key in state_dict and original_state_dict[key].shape == state_dict[key].shape:
                original_state_dict[key] = state_dict[key]
                matched += 1
        
        print(f'Loaded pretrained state_dict: {matched}/{total} matched')
        self.load_state_dict(original_state_dict)

    
class BaseNet(Base):
    """
    BaseNet inherits Base.
    BaseNet consists of Backbone(ResNet50), ASPP from DeepLabV3, Decoder., 
    """
    def __init__(self):
        super().__init__()

    def forward(self, src, bck):
        x = torch.cat([src, bck], dim=1)
        x0, x1, x2, x3, x4 = self.backbone(x)
        x4 = self.aspp(x4)
        x = self.decoder(x0, x1, x2, x3, x4)
        alp = x[:, 0:1].clamp_(0., 1.)
        fgr = x[:, 1:4].add(src).clamp_(0., 1.)
        err = x[:, 4:5].clamp_(0., 1.)
        hid = x[:, 5:].relu_()
        return alp, fgr, err, hid

class WholeNet(BaseNet)