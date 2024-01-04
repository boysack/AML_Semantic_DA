#!/usr/bin/python
# -*- encoding: utf-8 -*-
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
import torch
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
        
        self.root_samples = root_samples
        self.root_labels = root_labels
        self.transform = Compose([Resize((512, 1024), interpolation=Image.NEAREST), ToTensor(), Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
        self.samples = self._collect_samples()
        if max_iter is not None:
            self.samples = self.samples*(max_iter//len(self.samples) + 1) 
        

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img1 = Image.open(path).convert('RGB')
        img2 = Image.open(label)
        return self.transform(img1), 255*self.transform(img2)


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
