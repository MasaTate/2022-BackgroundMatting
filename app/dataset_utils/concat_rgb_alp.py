from torch.utils.data import Dataset

class ConcatRGBAlp(Dataset):
    def __init__(self, rgb_dataset, alp_dataset, transforms=None):
        self.transforms = transforms
        self.rgb_dataset = rgb_dataset
        self.alp_dataset = alp_dataset
        assert len(rgb_dataset) == len(alp_dataset)
    
    def __len__(self):
        return len(self.rgb_dataset)

    def __getitem__(self, index):
        rgb_alp = self.rgb_dataset[index], self.alp_dataset[index]
        if self.transforms:
            rgb_alp = self.transforms(*rgb_alp)
        return rgb_alp