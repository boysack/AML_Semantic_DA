from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from torchvision.transforms import v2# .v2 import Compose, ToTensor, Resize, Normalize, RandomCrop
import torchvision.transforms.functional as TF
import torch
import random
import numpy as np
import torchvision
from utils import FDA_source_to_target_np

#aprire un immagine GTA e una Cittyscapes e fare FDA

if __name__ == "__main__":
    cityscapes_path = Path(r"./Datasets/Cityscapes/Cityspaces/images/train/hanover/hanover_000000_000381_leftImg8bit.png")
    gta_path = Path(r"./Datasets/GTA5/images/train/00010.png")
    
    im_src = Image.open(gta_path).convert('RGB')
    im_trg = Image.open(cityscapes_path).convert('RGB')
    
    im_src = im_src.resize( (1024,512))
    im_trg = im_trg.resize( (1024,512))

    im_src = np.asarray(im_src, np.float32)
    im_trg = np.asarray(im_trg, np.float32)

    im_src = im_src.transpose((2, 0, 1))
    im_trg = im_trg.transpose((2, 0, 1))

    src_in_trg= im_src
    src_in_trg = FDA_source_to_target_np( im_src, im_trg, L=0.01 )

    src_in_trg = src_in_trg.transpose((1, 2, 0))
    src_in_trg = Image.fromarray(src_in_trg.astype(np.uint8))
    src_in_trg.save(f"./Images/FDA 0.01.png")

    im_src = im_src.transpose((1, 2, 0))
    im_src = Image.fromarray(im_src.astype(np.uint8))
    im_src.save(f"./Images/GTA.png")

    im_trg = im_trg.transpose((1, 2, 0))
    im_trg = Image.fromarray(im_trg.astype(np.uint8))
    im_trg.save(f"./Images/Cityscapes.png")

