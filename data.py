from datasets import load_dataset
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import numpy as np

_CFGS = {
    "mnist":   dict(hf="ylecun/mnist",    img_key="image", lbl_key="label", channels=1),
    "cifar10": dict(hf="uoft-cs/cifar10", img_key="img",   lbl_key="label", channels=3),
}


class ImageDataset(Dataset):
    def __init__(self, name: str, split: str = "train", img_size: int = 32):
        cfg = _CFGS[name]
        self.ds      = load_dataset(cfg["hf"], split=split)
        self.img_key = cfg["img_key"]
        self.lbl_key = cfg["lbl_key"]
        self.channels = cfg["channels"]

        mode = "L" if self.channels == 1 else "RGB"
        mean = [0.5] * self.channels
        std  = [0.5] * self.channels
        self.transform = T.Compose([
            T.Lambda(lambda x: x.convert(mode) if isinstance(x, Image.Image)
                               else Image.fromarray(np.asarray(x)).convert(mode)),
            T.Resize(img_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        item = self.ds[i]
        return self.transform(item[self.img_key]), item[self.lbl_key]
