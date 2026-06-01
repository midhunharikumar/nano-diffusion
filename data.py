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


class CachedImageDataset(Dataset):
    """Map-style dataset backed by a pre-resized uint8 tensor cache (see cache_data.py).

    Images live in RAM as uint8 [0,255]; __getitem__ normalizes to [-1,1] to match
    the ToTensor+Normalize([0.5],[0.5]) pipeline used elsewhere. Reading is a pure
    RAM op, so training is GPU-bound rather than network-bound.
    """

    def __init__(self, cache_file: str):
        blob = torch.load(cache_file, map_location="cpu", weights_only=True)
        self.images = blob["images"]   # uint8 (N, C, H, W)
        self.labels = blob["labels"]   # long  (N,)

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, i):
        img = self.images[i].float().div_(127.5).sub_(1.0)  # [0,255] → [-1,1]
        return img, int(self.labels[i])


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

    def _stream(self, worker, attempt: int):
        """Open a fresh HF stream. A new load_dataset builds a new httpx client,
        which is the only way to recover after a connection-reset closes the old one."""
        ds = load_dataset(self.hf_path, split=self.split, streaming=True)
        if self.shuffle_buffer > 0:
            base = worker.id if worker is not None else 0
            # vary seed per attempt so a restart doesn't replay the identical prefix
            ds = ds.shuffle(buffer_size=self.shuffle_buffer, seed=base + 1000 * attempt)
        it = iter(ds)
        if worker is not None:
            # Each worker takes every num_workers-th item starting at its own id
            it = itertools.islice(it, worker.id, None, worker.num_workers)
        return it

    def __iter__(self):
        worker  = torch.utils.data.get_worker_info()
        attempt = 0
        # HF streaming over flaky networks raises ConnectionReset / "client has been
        # closed" mid-epoch and kills the worker. Restart the stream on any such error
        # so training never dies on a transient hiccup (training-only; order is already
        # shuffled so a restart just resamples).
        while True:
            it = self._stream(worker, attempt)
            try:
                for item in it:
                    yield self.transform(item[self.img_key]), item[self.lbl_key]
                return  # stream exhausted cleanly → epoch done
            except Exception as e:  # noqa: BLE001 — deliberately broad: any I/O fault → reconnect
                attempt += 1
                if attempt > 100:
                    raise
                print(f"[stream w{getattr(worker, 'id', 0)}] reconnect #{attempt} after: "
                      f"{type(e).__name__}: {e}", flush=True)
