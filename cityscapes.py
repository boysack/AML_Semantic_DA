#!/usr/bin/python
# -*- encoding: utf-8 -*-
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Resize
# TODO


class CityScapes(Dataset):
    def __init__(self, mode):
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
        self.transform = Compose([Resize((512, 1024)), ToTensor()])
        self.samples = self._collect_samples()


    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img1 = Image.open(path)
        img2 = Image.open(label)
        return self.transform(img1), self.transform(img2)


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
