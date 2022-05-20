from typing import Tuple
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

    Output : Alpha, Foreground Residual, Refined Region
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
        self.conv4 = nn.Conv2d(channels[3], channels[4], kernel_size=3, bias=False)
        self.bn4 = nn.BatchNorm2d(channels[4])

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
        
        #upsample err, (hid, alp, fgr), (src, bck) to quater size of original resolution
        err = F.interpolate(err, (H_quat, W_quat), mode="bilinear", align_corners=False)
        x = F.interpolate(torch.cat([hid, alp, fgr], dim=1), (H_quat, W_quat), mode="bilinear", align_corners=False)
        y = F.interpolate(torch.cat([src, bck], dim=1), (H_quat, W_quat), mode="bilinear", align_corners=False)


        #select patches from err
        selected = self.select_pixel_to_refine(err)
        index = torch.nonzero(selected.squeeze(1)) #indices of non zero pixel (B, H, W). index.shape => torch.Size([number of patches,3])
        index = index[:,0], index[:,1], index[:,2]

        #crop patches from x
        x = self.crop_patches(x, index, 2, 3) #size=2,padding=3, that means 8*8 patches
        #crop patches from I and B
        y = self.crop_patches(y, index, 2, 3) #size=2,padding=3, that means 8*8 patches
        
        #pass conv & BN & ReLU
        x = self.conv1(torch.cat([x,y], dim=1))
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        #upsample x
        x = F.interpolate(x, 8, mode="nearest")
        
        #crop patches from I and B in original resolution
        y = self.crop_patches(torch.cat([src, bck], dim=1), index, 4, 2) #size=4,padding=2, that means 8*8 patches
        
        #pass conv & BN & ReLU
        x = self.conv3(torch.cat([x,y], dim=1))
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(x)

        #create out tensor & swap patches
        out = F.interpolate(torch.cat([alp, fgr], dim=1), (H_full, W_full), mode="bilinear", align_corners=False)
        out = self.swap_patches(x, out, index)

        alp = out[:, :1]
        fgr = out[:, 1:]

        return alp, fgr, selected



    def select_pixel_to_refine(self, err: torch.Tensor()):
        """
        Input : Error map

        Select top k patches to refine.

        Output : Selected pixel map. Binary map that includes only 0 or 1. 
        """
        #size
        err_b, _, err_h, err_w = err.shape
        #make err 2 dimentional
        err = err.view(err_b, -1)
        #indeces of top k pixel
        indices = err.topk(self.k, dim=1, sorted=False).indices
        #selected pixels in indeices is set to 1.
        selected = torch.zeros_like(err)
        selected = selected.scatter_(1, indices, 1.)
        selected = selected.view(err_b, 1, err_h, err_w)

        return selected

    def crop_patches(self,
                    image: torch.Tensor,
                    index: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                    size: int,
                    padding: int):
        """
        Input : image (B, C, H, W), index (Tuple[(P,...)]), size, padding

        1. patch height & width length is (size + 2 * padding)
        2. patch stride is (size) 
        3. crop all indexed patches from images
        """
        if padding > 0:
            #add padding to each side of height & width
            image = F.pad(image,(padding,padding,padding,padding))

        patch_size = size + padding * 2
        all_patches = image.permute(0,2,3,1).unfold(1,patch_size,size).unfold(2,patch_size,size)
        selected_patches = all_patches[index[0],index[1],index[2]]

        return selected_patches

    
    def swap_patches(self, x, out, index):
        """
        Input : Refined patches, Coarse mat, Patch indices

        1. Reshape out to torch.Tensor(B, number of height patches, number of width patches, channel, patch height, patch width)
        2. Swap x[idx[0], idx[1], idx[2]] in x
        3. Revise the shape of out
        
        Output : Swapped map 
        """
        #define size
        B_x, C_x, H_x, W_x = x.shape
        B_o, C_o, H_o, W_o = out.shape

        #Reshape out tensor
        out = out.view(B_o, C_o, H_o // H_x, H_x, W_o // W_x, W_x)
        out = out.permute(0, 2, 4, 1, 3, 5) 

         #Swap
        out[index[0], index[1], index[2]] = x

        #Revise shape of out tensor
        out = out.permute(0, 3, 1, 4, 2, 5).view(B_o, C_o, H_o, W_o)

        return out

        