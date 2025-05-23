from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision.transforms.functional as TF
import random
import numpy as np
import torch

class CamVidDataset(Dataset):
    def __init__(self, root, image_dir, mask_dir, crop_size=(360, 480), transform=True):
        self.image_paths = sorted(os.listdir(os.path.join(root, image_dir)))
        self.mask_paths = sorted(os.listdir(os.path.join(root, mask_dir)))
        self.image_dir = os.path.join(root, image_dir)
        self.mask_dir = os.path.join(root, mask_dir)
        self.crop_size = crop_size
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.image_dir, self.image_paths[idx])).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, self.mask_paths[idx]))

        if self.transform:
            i, j, h, w = TF.RandomCrop.get_params(img, output_size=self.crop_size)
            img = TF.crop(img, i, j, h, w)
            mask = TF.crop(mask, i, j, h, w)
            if random.random() > 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)

        img = TF.to_tensor(img)
        img = TF.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        mask = torch.as_tensor(np.array(mask), dtype=torch.long)
        return img, mask
