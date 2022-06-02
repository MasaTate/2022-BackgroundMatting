import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset_utils.image_dataset import ImageDataset
from dataset_utils.concat_img_bck import ConcatImgBck
import dataset_utils.augumentation as A
from torchvision import transforms as T
from torchvision.transforms.functional import to_pil_image
from models.model import WholeNet

def main(src_path, bck_path, pretrained_model, output_path, output_type):
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    print("device : "+str(device))

    #prepare dataset
    image_rgb_data = ImageDataset(src_path,"RGB") #source data
    image_bck_data = ImageDataset(bck_path,"RGB")#,transforms=T.RandomRotation(degrees=(180,180))) #background data
    image_rgb_bck = ConcatImgBck(image_rgb_data, image_bck_data, transforms=A.PairCompose([A.PairApply(nn.Identity()), A.PairApply(T.ToTensor())]))

    test_dataset = DataLoader(image_rgb_bck, batch_size=1, pin_memory=True)

    #prepare model
    model = WholeNet().to(device)
    model.eval()

    #load pretrained model
    pretrained_state_dict = torch.load(pretrained_model)
    matched , total = 0, 0
    original_state_dict = model.state_dict()
    for key in original_state_dict.keys():
        total +=1
        if key in pretrained_state_dict and original_state_dict[key].shape == pretrained_state_dict[key].shape:
            original_state_dict[key] = pretrained_state_dict[key]
            matched += 1
    model.load_state_dict(original_state_dict)
    print(f'Loaded pretrained state_dict: {matched}/{total} matched')

    #prepare output directory
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print("created output dir : "+output_path)
    print("========start inference=========")
    #inference
    with torch.no_grad():
        for i ,(src, bck) in enumerate(tqdm(test_dataset)):
            src = src.to(device)
            bck = bck.to(device)

            filename = image_rgb_bck.img_dataset.filenames[i]
            filename = os.path.relpath(filename, args.src_path)
            filename = os.path.splitext(filename)[0]

            alp, fgr, _, _, err, ref = model(src, bck)
            
            if 'com' in output_type:
                com = torch.cat([fgr * alp.ne(0), alp], dim=1)
                filepath = os.path.join(output_path, filename + '_com.png')
                com = to_pil_image(com[0])
                com.save(filepath)
                print("saved "+filepath)
            if 'alp' in output_type:
                filepath = os.path.join(output_path, filename + '_alp.jpg')
                alp = to_pil_image(alp[0])
                alp.save(filepath)
                print("saved "+filepath)
            if 'fgr' in output_type:
                filepath = os.path.join(output_path, filename + '_fgr.jpg')
                fgr = to_pil_image(fgr[0])
                fgr.save(filepath)
                print("saved "+filepath)
            if 'err' in output_type:
                err = F.interpolate(err, src.shape[2:], mode='bilinear', align_corners=False)
                filepath = os.path.join(output_path, filename + '_err.jpg')
                err = to_pil_image(err[0])
                err.save(filepath)
                print("saved "+filepath)
            if 'ref' in output_type:
                ref = F.interpolate(ref, src.shape[2:], mode='nearest')
                filepath = os.path.join(output_path, filename + '_ref.jpg')
                ref = to_pil_image(ref[0])
                ref.save(filepath)
                print("saved "+filepath)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("src_path", help="source directory path", type=str)
    parser.add_argument("bck_path", help="background directory path", type=str)
    parser.add_argument("output_type", help="choose output types from [composite layer, alpha matte, foreground residual, error map, reference map", nargs='*', choices=['com', 'alp', 'fgr', 'err', 'ref'])
    parser.add_argument("--pretrained_model", help="pretrained model path", type=str, default="../result/checkpoint/refinenet/checkpoint_epoch0_iter54999.pth")
    parser.add_argument("--output_path", help="output directory path", type=str, default="../result/images")
    

    args = parser.parse_args()

    main(args.src_path, args.bck_path, args.pretrained_model, args.output_path, args.output_type)