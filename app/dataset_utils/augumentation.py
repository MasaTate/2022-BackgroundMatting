from re import S
import torch
import numpy as np
import math
import random
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image, ImageFilter

class PairRandomAffineAndResize:
    def __init__(self, size, degrees, translate, scale, shear, ratio=(3./4., 4./3.), resample=Image.BILINEAR, fillcolor=0):
        self.size = size
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.ratio = ratio
        self.resample = resample
        self.fillcolor = fillcolor

    def __call__(self, *x):
        if not len(x):
            return

        width, height = x[0].size
        scale_factor = max(self.size[0] / height, self.size[1] / width)

        #chose bigger size between original/resized image
        height_pad = max(height, self.size[0])
        width_pad = max(width, self.size[1])

        #padding size
        pad_height = int(math.ceil((height_pad - height) / 2))
        pad_width = int(math.ceil((width_pad - width) / 2))

        #scale the "scale" & "translate" by scale_factor
        scale = self.scale[0] * scale_factor, self.scale[1] * scale_factor
        translate = self.translate[0] * scale_factor, self.translate[1] * scale_factor
        affine_params = T.RandomAffine.get_params(self.degrees, translate, scale, self.shear, (width,height))

        def transform(img):
            #if any padding is needed
            if pad_height > 0 or pad_width > 0:
                img = F.pad(img, (pad_height, pad_width))

            img = F.affine(img, *affine_params, self.resample, self.fillcolor)
            img = F.center_crop(img, self.size)
            return img

        return [transform(x_i) for x_i in x]

#for not pair
class RandomAffineAndResize(PairRandomAffineAndResize):
    def __call__(self, img):
        return super().__call__(img)[0]


class PairRandomHorizontalFlip(T.RandomHorizontalFlip):
    def __call__(self, *x):
        #the probability of flipping is p
        if torch.rand(1) < self.p:
            x = [F.hflip(x_i) for x_i in x]
        return x

class RandomBoxBlur:
    def __init__(self, prob, max_radius):
        self.prob = prob
        self.max_radius = max_radius

    def __call__(self, img):
        #the probability of bulurring is prob
        if torch.rand(1) < self.prob:
            filter = ImageFilter.BoxBlur(random.choice(range(self.max_radius+1)))
            img = img.filter(filter)
        return img

class PairRandomBoxBlur(RandomBoxBlur):
    def __call__(self, *x):
        #the probability of bulurring is prob
        if torch.rand(1) < self.prob:
            filter = ImageFilter.BoxBlur(random.choice(range(self.max_radius+1)))
            x = [x_i.filter(filter) for x_i in x]
        return x

