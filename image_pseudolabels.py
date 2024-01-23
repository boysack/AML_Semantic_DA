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
translation = {255: (0,0,0),0:(128,64,128), 1:(244, 35, 232), 2:(70,70,70), 3:(102,102,156), 5:(153,153,153), 6:(250,170,30), 8:(107,142,35), 9:(152,251,152), 10:(152,251,152), 13:(0,0,142) }
if __name__ == "__main__":
    gt_path = Path(r"./Datasets/Cityscapes/Cityspaces/gtFine/train/hanover/hanover_000000_000164_gtFine_color.png")
    gt_bw_path = Path(r"./Datasets/Cityscapes/Cityspaces/gtFine/train/hanover/hanover_000000_000164_gtFine_labelTrainIds.png")
    pseudolabel_path= Path(r"./Datasets/Cityscapes/Cityspaces/pseudolabels/hanover_000000_000164_leftImg8bit.png")

    img1 = Image.open(gt_path).convert('RGB')
    img1 = np.array(img1).astype(np.int32)
    img1 = Image.fromarray(img1.astype(np.uint8))
    img1.save(f"./Images/GT_Pseudolabel.png")

    img2 = Image.open(pseudolabel_path)
    img2 = np.array(img2).astype(np.int32)
    img2_color= np.zeros((img2.shape[0],img2.shape[1],3))
    for i in range(img2.shape[0]):
        for j in range(img2.shape[1]):
            img2_color[i,j,:]=translation[img2[i,j]]

    
    img2 = Image.fromarray(img2.astype(np.uint8))
    img2.save(f"./Images/BW_Pseudolabel.png")

    img2_color = Image.fromarray(img2_color.astype(np.uint8))
    img2_color.save(f"./Images/Color_Pseudolabel.png")


