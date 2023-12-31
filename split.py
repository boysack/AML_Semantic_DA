from pathlib import Path
from random import shuffle
import shutil

def split_80_20(p1:str, p2:str):
  a = Path(p1)
  b = Path(p2)

  images = [f for f in a.iterdir()]
  labels = [f for f in b.iterdir()] 

  samples = list(zip(images, labels))
  shuffle(samples)

  cut_p = round(len(samples) * .8)

  return samples[:cut_p], samples[cut_p:]

def move_all(samples: list, dest: str, mode: str):
  for sample in samples:
    shutil.move(sample[0], dest+"/images/"+mode)
    shutil.move(sample[1], dest+"/labels/"+mode)

train, val = split_80_20("./images", "./labels")
move_all(train, "./", "train")
move_all(train, "./", "val")