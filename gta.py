from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from torchvision.transforms import v2
import torchvision.transforms.functional as TF
import numpy as np
import torch
import augmentation
import random
import torchvision
# TODO

class GTA(Dataset):
    def __init__(self, mode, t=None, type=None):
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
        
        self.type=type
        self.mode = mode
        self.t = t
        self.transform = v2.Compose([v2.ToTensor(), v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

        #if t is not None:
        #    self.transform_train = v2.Compose([v2.RandomApply([augmentation.aug_transformations[t]], p = 0.5), v2.ToTensor(), v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
        #else:
        #self.transform_train = v2.Compose([v2.ToTensor(), v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
        
        # self.transform2 = Compose([Resize((1280, 720), interpolation=Image.NEAREST)])#'nearest-exact')])
        #self.transform_val = v2.Compose([v2.ToTensor(), v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

        self.transform = v2.Compose([v2.ToTensor(), v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
        self.tensor_transform = v2.Compose([v2.ToTensor()])
        self.samples = []
        self.samples += self._collect_samples()
  

    def __getitem__(self, idx):
        torchvision.disable_beta_transforms_warning()
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')
        label = Image.open(label)

        if self.mode == "train" or (self.mode == "all" and self.type is None):
            i, j, h, w = v2.RandomCrop.get_params(
                image, output_size=(720, 1280))
            image = TF.crop(image, i, j, h, w)
            label = TF.crop(label, i, j, h, w)

        if self.type == "FDA":
            '''i, j, h, w = v2.RandomCrop.get_params(
                image, output_size=(512, 1024))
            image = TF.crop(image, i, j, h, w)
            label = TF.crop(label, i, j, h, w)'''
            image = v2.Compose([v2.Resize((1024, 512))])(image)
            label = v2.Compose([v2.Resize((1024, 512))])(label)


        label = np.asarray(label, np.float32)

        label_copy = 255 * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v

        if self.t is not None and (self.mode == "train" or self.mode == "all"):
            if random.random() > 0.5:
                image = augmentation.aug_transformations[self.t](image)
                label = augmentation.label_transformations[self.t](label)
        
        if self.type != "FDA":
            image= self.transform(image)
        else:
            image = self.tensor_transform(image)

        return image, torch.tensor(label_copy.copy(), dtype=torch.float32)#dtype=torch.long)


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