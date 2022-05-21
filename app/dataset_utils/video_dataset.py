import cv2
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

class VideoDataset(Dataset):
    def __init__(self, path, transforms=None):
        self.cap = cv2.VideoCapture(path)
        self.transform = transforms
        self.num_frame = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.num_frame

    def __getitem__(self, index):
        """
        read frame of indices
        """
        if isinstance(index, slice):
            return[self[k] for k in range(*index.indices(len(self)))]

        if self.cap.get(cv2.CAP_PROP_POS_FRAMES) != index:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        #read frame from video
        ret, img = self.cap.read()
        #change channel order
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #convert numpy array to pil image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img