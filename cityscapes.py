#!/usr/bin/python
# -*- encoding: utf-8 -*-
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from torchvision.transforms import v2# .v2 import Compose, ToTensor, Resize, Normalize, RandomCrop
import torchvision.transforms.functional as TF
import torch
import random
import numpy as np
# TODO


class CityScapes(Dataset):
    def __init__(self, mode, max_iter=None):
        super(CityScapes, self).__init__()
        if mode == "train":
            root_samples = Path(r"./Datasets/Cityscapes/Cityspaces/images/train")
            root_labels = Path(r"./Datasets/Cityscapes/Cityspaces/gtFine/train")
            self.subdirs = ["hanover", "jena", "krefeld", "monchengladbach", "strasbourg", "stuttgart", "tubingen", "ulm", "weimar", "zurich"]
        elif mode == "val":
            root_samples = Path(r"./Datasets/Cityscapes/Cityspaces/images/val")
            root_labels = Path(r"./Datasets/Cityscapes/Cityspaces/gtFine/val")
            self.subdirs = ["frankfurt", "lindau", "munster"]
        else:
            raise Exception()
        
        self.mode = mode
        self.root_samples = root_samples
        self.root_labels = root_labels
        self.transform = v2.Compose([v2.ToTensor(), v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
        self.transform_label = v2.Compose([v2.ToTensor()])
        self.samples = self._collect_samples()
        if max_iter is not None:
            self.samples = self.samples*(max_iter//len(self.samples) + 1) 
        

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img1 = Image.open(path).convert('RGB')
        img2 = Image.open(label)

        if self.mode == "train": 
            image = img1.resize((1024, 512), Image.BICUBIC)
            label = img2.resize((1024, 512), Image.NEAREST)

        # convert into numpy array
        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.float32)

        # size = image.shape
        transforms = v2.Compose([
                v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
        image = transforms(image)
        image = image[:, :, ::-1]
        image = image.transpose((2, 0, 1))

        return image.copy(), label.copy()

        '''
        if self.mode == "train":
            #seed = random.random()
            #img1 = RandomCrop((512, 1024), seed=10, pad_if_needed=True)(img1)
            #img2 = RandomCrop((512, 1024), seed=10, pad_if_needed=True)(img2)
            i, j, h, w = v2.RandomCrop.get_params(
                img1, output_size=(512, 1024))
            img1 = TF.crop(img1, i, j, h, w)
            img2 = TF.crop(img2, i, j, h, w)

        #img2 = np.asarray(img2, np.float32)
        img2 = np.array(img2).astype(np.int64)[np.newaxis, :]
        return self.transform(img1), img2
        '''
        

    def __len__(self):
        return len(self.samples)


    def _collect_samples(self):
        samples = []
        labels = []

        for p in self.subdirs:
            samples += self._collect_imgs_sub_dir((self.root_samples / p), False)
            labels += self._collect_imgs_sub_dir((self.root_labels / p), True)

        samples = sorted(samples)
        labels = sorted(labels)

        return list(zip(samples, labels))


    @staticmethod
    def _collect_imgs_sub_dir(sub_dir: Path, val: bool):
        if not sub_dir.exists():
            raise ValueError(f"Data root must contain sub dir '{sub_dir.name}'")
        if val == False:
            return list(sub_dir.glob("*.png"))
        else:
            return list(sub_dir.glob("*Ids.png"))
