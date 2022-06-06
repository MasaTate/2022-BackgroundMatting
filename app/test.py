import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import transforms as T
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image
from PIL import Image
from dataset_utils.image_dataset import ImageDataset
from dataset_utils.concat_rgb_alp import ConcatRGBAlp
from dataset_utils.concat_img_bck import ConcatImgBck
from dataset_utils.small_dataset import SmallDataset
from torch.utils.tensorboard import SummaryWriter
import dataset_utils.augumentation as A
import argparse
import random
import kornia
import os
import time
from tqdm import tqdm
from models.model import WholeNet


def main(train_rgb_path,
        train_alp_path,
        train_bck_path,
        valid_rgb_path,
        valid_alp_path,
        valid_bck_path,
        batch_size,
        num_workers, 
        pretrained_model):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("device : "+str(device))

    #validation dataset
    print("preparing validation dataset...")
    valid_rgb_data = ImageDataset(valid_rgb_path,"RGB") #foreground data
    valid_alp_data = ImageDataset(valid_alp_path,"L") #alpha matte data
    valid_bck_data = ImageDataset(valid_bck_path,"RGB",transforms=T.Compose([
                                    A.RandomAffineAndResize((2048, 2048), degrees=(-5, 5), translate=(0.1, 0.1), scale=(1., 1.2), shear=(-5, 5)),
                                    T.ToTensor()
                                    ])) #background data
    valid_rbg_alp = ConcatRGBAlp(valid_rgb_data,valid_alp_data,
                                transforms=A.PairCompose([
                                    A.PairRandomAffineAndResize((2048, 2048), degrees=(-5, 5), translate=(0.1, 0.1), scale=(1., 1.2), shear=(-5, 5)),
                                    A.PairApply(T.ToTensor())
                                ])) #tuple of foreground & alpha matte with augumentation
    valid_rgb_alp_bck = ConcatImgBck(valid_rbg_alp, valid_bck_data) #tuple of foreground & alpha matte & background with augumentation

    valid_small_dataset = SmallDataset(valid_rgb_alp_bck, 4)
    valid_dataset = DataLoader(valid_small_dataset, pin_memory=True, batch_size=batch_size, num_workers=num_workers)

    #model
    print("setting up model...")
    model = WholeNet().to(device)
    model.load_state_dict(torch.load(pretrained_model))
    #load pretrained model
    """
    pretrained_state_dict = torch.load(pretrained_model)
    matched , total = 0, 0
    original_state_dict = model.state_dict()
    for key in original_state_dict.keys():
        total +=1
        if key in pretrained_state_dict and original_state_dict[key].shape == pretrained_state_dict[key].shape:
            original_state_dict[key] = pretrained_state_dict[key]
            matched += 1
    
    print(f'Loaded pretrained state_dict: {matched}/{total} matched')
    """

    output_path = "../result/images"
    writer = SummaryWriter("../logs/test")

    print("============start inference=============")

    def tensor_to_PIL(tensor):
        unloader = T.ToPILImage()
        image = tensor.cpu().clone()
        image = image.squeeze(0)
        image = unloader(image)
        return image

    #validation         
    model.eval()
    print("validating")
    with torch.no_grad():
        for i, ((fgr_in, alp_in), bck_in) in enumerate(valid_dataset):
            fgr_in = fgr_in.to(device)
            alp_in = alp_in.to(device)
            bck_in = bck_in.to(device)

            #composite foreground residual onto background
            src_in = fgr_in * alp_in + bck_in * (1 - alp_in)

            alp, fgr, _, _, err, _ = model(src_in, bck_in)

            writer.add_image('valid_alp_pred', make_grid(alp, nrow=5), i)
            writer.add_image('valid_fgr_pred', make_grid(fgr, nrow=5), i)
            writer.add_image('valid_com_pred', make_grid(fgr * alp, nrow=5), i)
            writer.add_image('valid_err_pred', make_grid(err, nrow=5), i)
            writer.add_image('valid_src_in', make_grid(src_in, nrow=5), i)
            writer.add_image('valid_bgr_in', make_grid(bck_in, nrow=5), i)
            """
            filename = valid_rgb_alp_bck.img_dataset[0].filenames[i]
            filename = os.path.relpath(filename, args.src_path)
            filename = os.path.splitext(filename)[0]

            filename = "test_"+str(i)
            
            filepath = os.path.join(output_path, filename + '_src.jpg')
            src_img = tensor_to_PIL(src_in)
            src_img.save(filepath)
            filepath = os.path.join(output_path, filename + '_bck.jpg')
            bck_img = tensor_to_PIL(bck_in)
            bck_img.save(filepath)

            com = torch.cat([fgr * alp.ne(0), alp], dim=1)
            filepath = os.path.join(output_path, filename + '_com.png')
            com = tensor_to_PIL(com)
            com.save(filepath)
            print("saved "+filepath)

            filepath = os.path.join(output_path, filename + '_alp.jpg')
            alp = tensor_to_PIL(alp)
            alp.save(filepath)
            print("saved "+filepath)
            """

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_rgb_path", help="foreground data path for training", type=str, default="../data/VideoMatte240K_JPEG_HD/train/fgr")
    parser.add_argument("--train_alp_path", help="alpha matte data path for training", type=str, default="../data/VideoMatte240K_JPEG_HD/train/pha")
    parser.add_argument("--train_bck_path", help="background data path for training", type=str, default="../data/Backgrounds/train")
    parser.add_argument("--valid_rgb_path", help="foreground data path for validation", type=str, default="../data/VideoMatte240K_JPEG_HD/test/fgr")
    parser.add_argument("--valid_alp_path", help="alpha matte data path for validation", type=str, default="../data/VideoMatte240K_JPEG_HD/test/pha")
    parser.add_argument("--valid_bck_path", help="background data path for validation", type=str, default="../data/Backgrounds/test")
    parser.add_argument("--batch_size", help="batch size", type=int, default=4)
    parser.add_argument("--num_workers", help="num workers", type=int, default=1)
    parser.add_argument("--pretrained_model", help="pretrained model path", type=str, default="../result/checkpoint/refinenet/checkpoint_epoch0_iter49.pth")
    args = parser.parse_args()

    dataset_path = "../data/VideoMatte240K_JPEG_HD/train/fgr"
    alpha_path = "../data/VideoMatte240K_JPEG_HD/train/pha"
    background_path = "../data/Backgrounds"


    arg = parser.parse_args()
    main(args.train_rgb_path,
        args.train_alp_path,
        args.train_bck_path,
        args.valid_rgb_path,
        args.valid_alp_path,
        args.valid_bck_path,
        args.batch_size,
        args.num_workers, 
        args.pretrained_model, 
        )