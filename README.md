# 2022-BackgroundMatting
This repository is for `Real-Time High-Resolution Background Matting, CVPR 2021`([pdf](https://openaccess.thecvf.com/content/CVPR2021/papers/Lin_Real-Time_High-Resolution_Background_Matting_CVPR_2021_paper.pdf)) reimplementation.  
I referred to the [official implementaion](https://github.com/PeterL1n/BackgroundMattingV2) in PyTorch.  
I used pretrained weights of DeepLabV3 from [VainF](https://github.com/VainF/DeepLabV3Plus-Pytorch).

## Requirements
I share anaconda environment yml file.
Create environment by `conda env create -n $ENV_NAME -f py38torch1110.yml`  
You can also check requirements from the yml file.


## Usage
### Training Base Network
The Base Network includes ASPP module from DeepLabV3. I used pretrained DeepLabV3 weight([best_deeplabv3_resnet50_voc_os16.pth](https://www.dropbox.com/sh/w3z9z8lqpi8b2w7/AAB0vkl4F5vy6HdIhmRCTKHSa?dl=0)).

```
usage: train_base.py [-h] [--train_rgb_path TRAIN_RGB_PATH] [--train_alp_path TRAIN_ALP_PATH] [--train_bck_path TRAIN_BCK_PATH] [--valid_rgb_path VALID_RGB_PATH] [--valid_alp_path VALID_ALP_PATH]
                     [--valid_bck_path VALID_BCK_PATH] [--checkpoint_path CHECKPOINT_PATH] [--logging_path LOGGING_PATH] [--batch_size BATCH_SIZE] [--num_workers NUM_WORKERS] [--pretrained_model PRETRAINED_MODEL]
                     --epochs EPOCHS

optional arguments:
  -h, --help            show this help message and exit
  --train_rgb_path TRAIN_RGB_PATH
                        foreground data directory path for training
  --train_alp_path TRAIN_ALP_PATH
                        alpha matte data directory path for training
  --train_bck_path TRAIN_BCK_PATH
                        background data directory path for training
  --valid_rgb_path VALID_RGB_PATH
                        foreground data directory path for validation
  --valid_alp_path VALID_ALP_PATH
                        alpha matte data directory path for validation
  --valid_bck_path VALID_BCK_PATH
                        background data directory path for validation
  --checkpoint_path CHECKPOINT_PATH
                        checkpoint saving dir path
  --logging_path LOGGING_PATH
                        path to save logs
  --batch_size BATCH_SIZE
                        batch size
  --num_workers NUM_WORKERS
                        num workers
  --pretrained_model PRETRAINED_MODEL
                        pretrained model path
  --epochs EPOCHS       epochs to train
```

### Training Whole Network (Refinement Network)
After training the Base Network, train the Base Network and Refinement Network jointly.  
```
usage: train_refine.py [-h] [--train_rgb_path TRAIN_RGB_PATH] [--train_alp_path TRAIN_ALP_PATH] [--train_bck_path TRAIN_BCK_PATH] [--valid_rgb_path VALID_RGB_PATH] [--valid_alp_path VALID_ALP_PATH]
                       [--valid_bck_path VALID_BCK_PATH] [--checkpoint_path CHECKPOINT_PATH] [--logging_path LOGGING_PATH] [--batch_size BATCH_SIZE] [--num_workers NUM_WORKERS] [--pretrained_model PRETRAINED_MODEL]
                       --epochs EPOCHS

optional arguments:
  -h, --help            show this help message and exit
  --train_rgb_path TRAIN_RGB_PATH
                        foreground data directory path for training
  --train_alp_path TRAIN_ALP_PATH
                        alpha matte data directory path for training
  --train_bck_path TRAIN_BCK_PATH
                        background data directory path for training
  --valid_rgb_path VALID_RGB_PATH
                        foreground data directory path for validation
  --valid_alp_path VALID_ALP_PATH
                        alpha matte data directory path for validation
  --valid_bck_path VALID_BCK_PATH
                        background data directory path for validation
  --checkpoint_path CHECKPOINT_PATH
                        checkpoint saving dir path
  --logging_path LOGGING_PATH
                        path to save logs
  --batch_size BATCH_SIZE
                        batch size
  --num_workers NUM_WORKERS
                        num workers
  --pretrained_model PRETRAINED_MODEL
                        pretrained model path
  --epochs EPOCHS       epochs to train
  ```

### Test Image Background Matting
You can download my trained weight form [here](https://drive.google.com/drive/folders/1UnoNk7fp44PyDsyfdnIc6-wAzNxP9xgn?usp=sharing).  
Using trained weight, you can test image background matting.  
Make sure that related image and background data are same order in each directory.
```
usage: test_image.py [-h] [--pretrained_model PRETRAINED_MODEL] [--output_path OUTPUT_PATH] src_path bck_path [{com,alp,fgr,err,ref} [{com,alp,fgr,err,ref} ...]]

positional arguments:
  src_path              source directory path
  bck_path              background directory path
  {com,alp,fgr,err,ref}
                        choose output types from [composite layer, alpha matte, foreground residual, error map, reference map

optional arguments:
  -h, --help            show this help message and exit
  --pretrained_model PRETRAINED_MODEL
                        pretrained model path
  --output_path OUTPUT_PATH
                        output directory path
```

## Datasets
Limited datasets are available on the [official website](https://grail.cs.washington.edu/projects/background-matting-v2/#/datasets).

## Examples


## References
- S.Lin, A.Ryabtsev, S.Sengupta, B.Curless, S.Seitz, I.Kemelmacher-Shlizerman. "Real-Time High-Resolution Background Matting.", in CVPR, 2021. ([pdf](https://openaccess.thecvf.com/content/CVPR2021/papers/Lin_Real-Time_High-Resolution_Background_Matting_CVPR_2021_paper.pdf))
- [Official Home Page](https://grail.cs.washington.edu/projects/background-matting-v2/#/)
- [Official implementation in PyTorch](https://github.com/PeterL1n/BackgroundMattingV2)
- [DeepLabV3 pretrained weights](https://github.com/VainF/DeepLabV3Plus-Pytorch)
- L.C.Chen, G.Papandreou, F.Schroff, H.Adam. "Rethinking Atrous Convolution for Semantic Image Segmentation.", in CVPR 2017. ([arxiv](https://arxiv.org/abs/1706.05587))
