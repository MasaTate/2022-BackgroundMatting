from email import parser
import resnet
import decoder
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import glob
import argparse
from torchvision.models.segmentation.deeplabv3 import ASPP

loader = transforms.Compose([transforms.ToTensor()])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def main(content_path):
    img_pths = glob.glob(content_path + "/*.png")
    print("device : ")
    print(device)
    content_img = Image.open(img_pths[0])
    content_img = content_img.convert("RGB")
    tensor_img = loader(content_img).unsqueeze(0).to(device)
    """
    for n in range(1,2):
        add_img = Image.open(img_pths[n])
        add_img = content_img.convert("RGB")
        tensor_img = torch.cat([tensor_img,loader(add_img).unsqueeze(0).to(device)],dim=0)
    """
    print(tensor_img.shape)

    model = resnet.ResNet50()
    model.to(device).eval()
    x0,x1,x2,x3,x4 = model(tensor_img)
    aspp = ASPP(2048, [3, 6, 9])
    aspp.eval()
    print(x4.shape)
    x = aspp(x4)
    print(x.shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--content_path",type=str,help="content dir path")
    arg = parser.parse_args()
    main(arg.content_path)