import numpy as np
from PIL import Image
import lzma
import os
import pickle
from pathlib import Path
import tqdm

in_folder = Path("data_images")
out_folder = Path("data_images_rescaled")

out_folder.mkdir(exist_ok=True, parents=True)

SIZE = 256

def rescale(img: np.ndarray):
    size: int = min(img.shape[0], img.shape[1])
    top: int = (img.shape[0] - size) // 2
    left: int = (img.shape[1] - size) // 2
    cropped: np.ndarray = img[top:top+size, left:left+size]
    image: Image.Image = Image.fromarray(cropped)
    return np.array(image.resize((SIZE, SIZE)))


for filename in tqdm.tqdm(os.listdir(in_folder)):
    if not filename.endswith(".npz"):
        continue
    
    inpath = in_folder / filename
    outpath = out_folder / filename

    with lzma.open(inpath, "rb") as file, lzma.open(outpath, "wb") as out:
        data = pickle.load(file)
        for i, frame in tqdm.tqdm(enumerate(data), unit="frame"):
            frame.image = rescale(frame.image)
        pickle.dump(data, out)