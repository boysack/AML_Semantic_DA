from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
import random
# TODO


class GTA(Dataset):
    def __init__(self):
        super(GTA, self).__init__()
        self.root_samples = Path(r"./Datasets/GTA5/images")
        self.root_labels = Path(r"./Datasets/GTA5/labels")
        self.transform = Compose([Resize((512, 1024), interpolation=Image.NEAREST), ToTensor()])
        self.samples = self._collect_samples()
        

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img1 = Image.open(path)
        img2 = Image.open(label)  ## < qui sono 3 canali, gestire
        return self.transform(img1), self.transform(img2)


    def __len__(self):
        return len(self.samples)


    def _collect_samples(self):
        """Collect all paths and labels
        
        Helper method for the constructor
        """
        sample_path += self._collect_imgs_sub_dir(self.root_samples)
        label_path += self._collect_imgs_sub_dir(self.root_labels)

        sample_path = sorted(sample_path)
        label_path = sorted(label_path)

        samples = list(zip(sample_path, label_path))
        random.shuffle(samples)


    @staticmethod
    def _collect_imgs_sub_dir(sub_dir: Path):
        if not sub_dir.exists():
            raise ValueError(f"Data root must contain sub dir '{sub_dir.name}'")
        return list(sub_dir.glob("*.png"))