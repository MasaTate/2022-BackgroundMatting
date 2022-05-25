from dataset_utils.image_dataset import ImageDataset
from dataset_utils.concat_rgb_alp import ConcatRGBAlp
from dataset_utils.concat_img_bck import ConcatImgBck
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import argparse

def main(dataset_path,alpha_path,background_path):
    train_rgb_data = ImageDataset(dataset_path,"RGB") #foreground data
    train_alp_data = ImageDataset(alpha_path,"L") #alpha matte data
    train_bck_data = ImageDataset(background_path,"RGB") #background data
    train_rbg_alp = ConcatRGBAlp(train_rgb_data,train_alp_data) #tuple of foreground & alpha matte
    train_rgb_alp_bck = ConcatImgBck(train_rbg_alp, train_bck_data) #tuple of foreground & alpha matte & background
    print(train_rgb_data.__len__())
    print(train_alp_data.__len__())
    print(train_rgb_alp_bck[0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path")
    parser.add_argument("--alpha_path")
    parser.add_argument("--background_path")
    dataset_path = "../data/VideoMatte240K_JPEG_HD/train/fgr"
    alpha_path = "../data/VideoMatte240K_JPEG_HD/train/pha"
    background_path = "../data/Backgrounds"

    arg = parser.parse_args()
    main(dataset_path,alpha_path,background_path)