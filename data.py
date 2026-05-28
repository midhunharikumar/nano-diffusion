import itertools

from datasets import load_dataset
from torch.utils.data import Dataset
import torch
from PIL import Image
import torchvision.transforms as T
import numpy as np


class _ConvertMode:
    """Picklable replacement for T.Lambda(convert); needed for num_workers > 0 on spawn."""
    def __init__(self, mode: str):
        self.mode = mode

    def __call__(self, x):
        if isinstance(x, Image.Image):
            return x.convert(self.mode)
        return Image.fromarray(np.asarray(x)).convert(self.mode)


_CFGS = {
    "mnist":    dict(hf="ylecun/mnist",       img_key="image", lbl_key="label", channels=1,
                     size=60000),
    "cifar10":  dict(hf="uoft-cs/cifar10",    img_key="img",   lbl_key="label", channels=3,
                     size=50000),
    # gated dataset — requires HuggingFace token with accepted terms
    "imagenet": dict(hf="ILSVRC/imagenet-1k", img_key="image", lbl_key="label", channels=3,
                     eval_split="validation",  size=1281167),
    # pre-resized to 256×256, ungated, no HF token required
    "imagenet_256x256": dict(hf="benjamin-paine/imagenet-1k-256x256", img_key="image",
                             lbl_key="label", channels=3,
                             eval_split="validation", size=1281167),
}
# Resolution-specific aliases that map to the same HF dataset.
# Allows `prepare_data.py imagenet64` to work when DATASET matches the config filename.
_CFGS["imagenet64"]  = _CFGS["imagenet_256x256"]
_CFGS["imagenet256"] = _CFGS["imagenet_256x256"]


def _make_transform(channels: int, img_size: int) -> T.Compose:
    mode = "L" if channels == 1 else "RGB"
    return T.Compose([
        _ConvertMode(mode),
        T.Resize(img_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize([0.5] * channels, [0.5] * channels),
    ])


class ImageDataset(Dataset):
    """Map-style dataset — downloads and caches the full split before training."""

    def __init__(self, name: str, split: str = "train", img_size: int = 32):
        cfg = _CFGS[name]
        self.ds        = load_dataset(cfg["hf"], split=split)
        self.img_key   = cfg["img_key"]
        self.lbl_key   = cfg["lbl_key"]
        self.transform = _make_transform(cfg["channels"], img_size)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        item = self.ds[i]
        return self.transform(item[self.img_key]), item[self.lbl_key]


class StreamingImageDataset(torch.utils.data.IterableDataset):
    """
    Streams shards from HuggingFace on demand — no full dataset download required.

    Each DataLoader worker opens its own stream and takes every num_workers-th item
    (round-robin sharding), so the combined output is the full dataset once per epoch.
    Shuffling is buffer-based (per worker, independent seeds).
    """

    def __init__(self, name: str, split: str = "train", img_size: int = 32,
                 shuffle_buffer: int = 1000):
        cfg = _CFGS[name]
        self.hf_path        = cfg["hf"]
        self.split          = split
        self.img_key        = cfg["img_key"]
        self.lbl_key        = cfg["lbl_key"]
        self.shuffle_buffer = shuffle_buffer
        self.transform      = _make_transform(cfg["channels"], img_size)

    def __iter__(self):
        worker      = torch.utils.data.get_worker_info()
        ds          = load_dataset(self.hf_path, split=self.split, streaming=True)
        if self.shuffle_buffer > 0:
            seed = worker.id if worker is not None else 0
            ds   = ds.shuffle(buffer_size=self.shuffle_buffer, seed=seed)
        it = iter(ds)
        if worker is not None:
            # Each worker takes every num_workers-th item starting at its own id
            it = itertools.islice(it, worker.id, None, worker.num_workers)
        for item in it:
            yield self.transform(item[self.img_key]), item[self.lbl_key]
