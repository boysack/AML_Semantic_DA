from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, RandomApply
import random
import numpy as np
import torch
import augmentation
# TODO


class GTA(Dataset):
    def __init__(self, mode, t):
        super(GTA, self).__init__()
        if mode == "train":
            self.root_samples = [Path(r"./Datasets/GTA5/images/train")]
            self.root_labels = [Path(r"./Datasets/GTA5/labels/train")]
        elif mode == "val":
            self.root_samples = [Path(r"./Datasets/GTA5/images/val")]
            self.root_labels = [Path(r"./Datasets/GTA5/labels/val")]
        elif mode == "all":
            self.root_samples = [Path(r"./Datasets/GTA5/images/train"), Path(r"./Datasets/GTA5/images/val")]
            self.root_labels = [Path(r"./Datasets/GTA5/labels/train"), Path(r"./Datasets/GTA5/labels/val")]
        else:
            raise Exception()
        
        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}
        if t is not None:
            self.transform1 = Compose([Resize((512, 1024), interpolation=Image.NEAREST), RandomApply([augmentation.aug_transformations[t]], p = 0.5), ToTensor()])
        else:
            self.transform1 = Compose([Resize((512, 1024), interpolation=Image.NEAREST), ToTensor()])
        
        self.transform2 = Compose([Resize((512, 1024), interpolation=Image.NEAREST)])
        
        self.samples = []
        self.samples += self._collect_samples()
  

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img1 = Image.open(path)
        img2 = Image.open(label) 
        img2 = self.transform2(img2)

        img2 = np.asarray(img2, np.float32)

        label_copy = 255 * np.ones(img2.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[img2 == k] = v

        return self.transform1(img1), torch.tensor(label_copy.copy(), dtype=torch.long)


    def __len__(self):
        return len(self.samples)


    def _collect_samples(self):
        """Collect all paths and labels
        
        Helper method for the constructor
        """
        samples = []
        labels = []

        for root in self.root_samples:
            samples += self._collect_imgs_sub_dir(root)
            
        for root in self.root_labels:
            labels += self._collect_imgs_sub_dir(root)

        samples = sorted(samples)
        labels = sorted(labels)

        return list(zip(samples, labels))


    @staticmethod
    def _collect_imgs_sub_dir(sub_dir: Path):
        if not sub_dir.exists():
            raise ValueError(f"Data root must contain sub dir '{sub_dir.name}'")
        return list(sub_dir.glob("*.png"))