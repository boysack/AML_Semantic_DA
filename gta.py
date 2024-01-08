from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, RandomApply, Normalize
import random
import numpy as np
import torch
import augmentation
# TODO

class GTA(Dataset):
    def __init__(self, mode, t=None):
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
        
        self.mode = mode
        
        '''self.id_to_trainid = {(128, 64, 128): 0, (244, 35, 232): 1, (70, 70, 70): 2, (102, 102, 156): 3,
                              (190, 153, 153): 4, (153, 153, 153): 5, (250, 170, 30): 6, (220, 220, 0): 7,
                              (107, 142, 35): 8, (152, 251, 152): 9, (70, 130, 180): 10, (220, 20, 60): 11,
                              (255, 0, 0): 12, (0, 0, 142): 13, (0, 0, 70): 14, (0, 60, 100): 15,
                              (0, 80, 100): 16, (0, 0, 230): 17, (119, 11, 32): 18, (0, 0, 0): 255}'''
        
        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

        if t is not None:
            self.transform1 = Compose([Resize((1280, 720), interpolation=Image.NEAREST), RandomApply([augmentation.aug_transformations[t]], p = 0.5), ToTensor(), Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
        else:
            self.transform1 = Compose([Resize((1280, 720), interpolation=Image.NEAREST), ToTensor(), Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
        
        self.transform2 = Compose([Resize((1280, 720), interpolation=Image.NEAREST)])#'nearest-exact')])
        self.transform_val = Compose([ToTensor(), Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

        self.samples = []
        self.samples += self._collect_samples()
  

    '''def __getitem__(self, idx):
        path, label = self.samples[idx]
        img1 = Image.open(path).convert('RGB')
        img2 = Image.open(label).convert('RGB')
        print(img2)
        # img2 = self.transform2(img2)

        img2 = np.asarray(img2, np.int32)
        print(img2.shape)
        print(set([tuple(img2[i][j]) for i in range(img2.shape[0]) for j in range(img2.shape[1])]))

        label_copy = 255 * np.ones(img2.shape, dtype=np.int32)
        for row in range(img2.shape[0]):
            for col in range(img2.shape[1]):
                label_copy[row][col] = self.id_to_trainid[tuple(img2[row][col])]


        return self.transform1(img1), torch.tensor(label_copy.copy(), dtype=torch.long)'''
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img1 = Image.open(path).convert('RGB') # << added here
        img2 = Image.open(label).convert('RGB') # << added here
        img2 = self.transform2(img2)

        img2 = np.asarray(img2, np.float32)

        label_copy = 255 * np.ones(img2.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[img2 == k] = v

        return self.transform1(img1) if self.mode=="train" else self.transform_val(img1), torch.tensor(label_copy.copy(), dtype=torch.long)


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