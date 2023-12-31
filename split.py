from pathlib import Path
from random import shuffle
import shutil
import os
from math import floor

def split_80_20(p1:str, p2:str):
  a = Path(p1)
  b = Path(p2)

  images = [f for f in a.iterdir()]
  labels = [f for f in b.iterdir()] # sia colorate che greyscale?

  samples = list(zip(images, labels))
  shuffle(samples)

  cut_p = floor(len(samples) * .8)

  return samples[:cut_p], samples[cut_p:]

def move_all(samples: list, dest: str, mode: str):
  os.mkdir(dest+"images/"+mode)
  os.mkdir(dest+"labels/"+mode)
  for sample in samples:
    shutil.move(sample[0], dest+"images/"+mode)
    shutil.move(sample[1], dest+"labels/"+mode)

if __name__=="__main__":
  train, val = split_80_20("./images", "./labels")
  move_all(train, "./", "train")
  move_all(val, "./", "val")