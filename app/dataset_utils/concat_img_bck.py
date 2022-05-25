from torch.utils.data import Dataset

class ConcatImgBck(Dataset):
    def __init__(self, img_dataset, bck_dataset, transforms=None):
        self.transforms = transforms
        self.img_dataset = img_dataset
        self.bck_dataset = bck_dataset
    
    def __len__(self):
        return len(max(len(self.img_dataset),len(self.bck_dataset)))

    def __getitem__(self, index):
        img_bck = self.img_dataset[index%len(self.img_dataset)], self.bck_dataset[index%len(self.bck_dataset)]
        if self.transforms:
            img_bck = self.transforms(*img_bck)
        return img_bck