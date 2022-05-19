from statistics import mode
import torch
import torch.nn as nn
import torch.nn.functional as F

class Refine(nn.Module):
    """
    Input : Alpha(alp), Foreground Residual(fgr), Hidden(hid), Source(src), Background(bck)
    
    1. resize alp,fgr,hid,src,bck to H/2*W/2
    2. select patches to refine
    3. pass patches (3*3 conv , batch normalization, relu) * 2 -> 4*4 patch
    4. upsample 3 to 8*8 & concatenate 8*8 patches of I, B
    5. redo no.3 -> 4*4 alp & fgr come out
    6. upsample alp & fgr to original resolution
    7. swap in refined patches

    Output : Alpha, Foreground Residual
    """
    def __init__(self, k_patches: int):
        super().__init__()

        # refine k patches
        self.k = k_patches

        channels = [32, 24, 16, 12, 4]
        #Input : Alpha, Foreground Residual, Hidden, I, B -> channel = 1 + 3 + 32 + 3 + 3 
        self.conv1 = nn.Conv2d(1+channels[0]+3+6, channels[1], kernel_size=3, bias=False)
        self.bn1 = nn.BatchNorm2d(channels[1])
        self.conv2 = nn.Conv2d(channels[1], channels[2], kernel_size=3, bias=False)
        self.bn2 = nn.BatchNorm2d(channels[2])

        #Concatenate 4*4 patches and I, B patches -> channels[2] + 3 + 3
        self.conv3 = nn.Conv2d(1+channels[2]+6, channels[3], kernel_size=3, bias=False)
        self.bn3 = nn.BatchNorm2d(channels[3])
        self.conv2 = nn.Conv2d(channels[3], channels[4], kernel_size=3, bias=False)
        self.bn2 = nn.BatchNorm2d(channels[4])

        self.relu = nn.ReLU(True)

    def forward(self, 
                src: torch.Tensor(), #size = (B, 3, H, W)
                bck: torch.Tensor(), #size = (B, 3, H, W)
                alp: torch.Tensor(), #size = (B, 1, H/c, W/c)
                fgr: torch.Tensor(), #size = (B, 3, H/c, W/c)
                err: torch.Tensor(), #size = (B, 3, H/c, W/c)
                hid: torch.Tensor()  #size = (B, 3, H/c, W/c)
                ):

        #size definition        
        H_full = src.shape[2]
        W_full = src.shape[3]
        H_half = H_full // 2
        W_half = W_full // 2
        H_quat = H_full // 4
        W_quat = W_full // 4

        err = F.interpolate(err, (H_quat, W_quat), mode="bilinear", align_corners=False)
        x = F.interpolate(torch.cat([alp, fgr, hid], (H_quat, W_quat), mode="bilinear", align_corners=False))

        selected = self.select_pixel_to_refine(err)

    def select_pixel_to_refine(self, err: torch.Tensor()):
        """
        Input : Error map

        Select top k patches to refine.

        Output : Selected pixel map. Binary map that includes only 0 or 1. 
        """
        #size
        err_b, _, err_h, err_w = err.shape
        err = err.view(err_b, -1)
        indices = err.topk(self.k, dim=1, sorted=False).indices
        selected = torch.zeros_like(err)
        selected = selected.scatter_(1, indices, 1.)
        selected = selected.view(err_b, 1, err_h, err_w)

        return selected

    