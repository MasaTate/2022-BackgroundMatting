from email import parser
import resnet
import decoder
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import argparse
from torchvision.models.segmentation.deeplabv3 import ASPP

loader = transforms.Compose([transforms.ToTensor()])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def main(content_path):
    print("device : ")
    print(device)
    content_img = Image.open(content_path)
    content_img = content_img.convert("RGB")
    tensor_img = loader(content_img).unsqueeze(0).to(device)
    print(tensor_img.shape)

    model = resnet.ResNet50()
    x0,x1,x2,x3,x4 = model(tensor_img)
    aspp = ASPP(2048,[3,6,9])
    x4 = aspp(x4)
    print(x4.shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--content_img",type=str,help="content image path")

    arg = parser.parse_args()
    main(arg.content_img)