import os
import glob
from torch.utils.data import Dataset
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, root_pth, mode='RGB', transforms=None):
        self.transforms = transforms
        self.mode = mode
        self.filenames = sorted([*glob.glob(os.path.join(root_pth, '**', '*.jpg'), recursive=True),
                                    *glob.glob(os.path.join(root_pth, '**', '*.png'), recursive=True)])
    
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        with Image.open(self.filenames[index]) as img:
            img = img.convert(self.mode)

        if self.transforms: img = self.transforms(img)

        return img