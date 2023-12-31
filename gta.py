from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Grayscale
import random
# TODO


class GTA(Dataset):
    def __init__(self, mode):
        super(GTA, self).__init__()
        if mode == "train":
            root_samples = Path(r"./Datasets/GTA5/images/train")
            root_labels = Path(r"./Datasets/GTA5/labels/train")
        elif mode == "val":
            root_samples = Path(r"./Datasets/GTA5/images/val")
            root_labels = Path(r"./Datasets/GTA5/labels/val")
        else:
            raise Exception()
        
        self.root_samples = root_samples
        self.root_labels = root_labels
        self.transform1 = Compose([Resize((512, 1024), interpolation=Image.NEAREST), ToTensor()])
        self.transform2 = Compose([Resize((512, 1024), interpolation=Image.NEAREST), Grayscale(), ToTensor()])
        self.samples = self._collect_samples()
        

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img1 = Image.open(path)
        img2 = Image.open(label) 
        return self.transform1(img1), 255*self.transform2(img2)


    def __len__(self):
        return len(self.samples)


    def _collect_samples(self):
        """Collect all paths and labels
        
        Helper method for the constructor
        """
        samples = []
        labels = []

        samples += self._collect_imgs_sub_dir(self.root_samples)
        labels += self._collect_imgs_sub_dir(self.root_labels)

        samples = sorted(samples)
        labels = sorted(labels)

        return list(zip(samples, labels))


    @staticmethod
    def _collect_imgs_sub_dir(sub_dir: Path):
        if not sub_dir.exists():
            raise ValueError(f"Data root must contain sub dir '{sub_dir.name}'")
        return list(sub_dir.glob("*.png"))