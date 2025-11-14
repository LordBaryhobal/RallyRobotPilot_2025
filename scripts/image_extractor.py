import lzma
from pathlib import Path
import pickle

from PIL import Image
import numpy as np

from rallyrobopilot.sensing_message import SensingSnapshot


def main():
    dir = Path("/tmp/images")
    dir.mkdir(exist_ok=True)
    with lzma.open("generated/SimpleTrack_0.npz", "rb") as f:
        data: list[SensingSnapshot] = pickle.load(f)

        for i, snap in enumerate(data):
            img: np.ndarray = snap.image # type: ignore
            im: Image.Image = Image.fromarray(img)
            im.save(dir / f"{i:03d}.jpg")


if __name__ == "__main__":
    main()
