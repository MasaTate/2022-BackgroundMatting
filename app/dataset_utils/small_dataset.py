from torch.utils.data import Dataset

#thin out original dataset
class SmallDataset(Dataset):
    def __init__(self, dataset, num):
        num = min(num, len(dataset))
        self.dataset = dataset
        self.indices = []
        for i in range(num):
            self.indices.append(i * int(len(dataset) / num))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]