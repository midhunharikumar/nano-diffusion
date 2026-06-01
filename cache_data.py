"""
Build a pre-resized in-RAM tensor cache of N ImageNet shards for fast training.

The HF streaming pipeline is data-bound: step count (and thus FID in a fixed-time
run) is throttled by network/decode of 256px parquets, with high variance across
concurrent runs. This downloads whole shards once (bulk file fetch, not per-item
HTTP), decodes + resizes to img_size, and stores a uint8 tensor on the persistent
volume. Training then reads from RAM — GPU-bound, deterministic, no network.

One-time cost (download + decode) runs before the training timer starts and is
cached on the volume, so subsequent runs load it instantly.
"""
import os

import torch
import torchvision.transforms as T

from data import _CFGS, _ConvertMode


def build_cache(dataset: str, img_size: int, n_shards: int, out_path: str,
                n_images: int = 0) -> str:
    """Download n_shards parquet shards, decode+resize to img_size, save uint8 tensor.

    n_images=0 → use every image in the downloaded shards.
    Returns out_path. No-op (just returns) if the cache already exists.
    """
    if os.path.exists(out_path):
        print(f"[cache] reuse {out_path}", flush=True)
        return out_path

    from datasets import Image as HFImage
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    cfg = _CFGS[dataset]
    repo, img_key, lbl_key, ch = (cfg["hf"], cfg["img_key"], cfg["lbl_key"], cfg["channels"])
    mode = "L" if ch == 1 else "RGB"
    # transform stops at PILToTensor → uint8 [0,255] CHW; normalization to [-1,1]
    # happens at read time in CachedImageDataset (keeps the cache compact).
    tf = T.Compose([
        _ConvertMode(mode),
        T.Resize(img_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(img_size),
        T.PILToTensor(),
    ])

    files = []
    for i in range(n_shards):
        fn = f"data/train-{i:05d}-of-00040.parquet"
        print(f"[cache] downloading {fn}", flush=True)
        files.append(hf_hub_download(repo, fn, repo_type="dataset"))

    ds = load_dataset("parquet", data_files=files, split="train")
    # raw parquet stores the image as a {bytes,path} struct — cast so it decodes to PIL
    try:
        ds = ds.cast_column(img_key, HFImage())
    except Exception as e:  # noqa: BLE001
        print(f"[cache] cast_column skipped: {e}", flush=True)

    total = len(ds) if n_images <= 0 else min(n_images, len(ds))
    imgs = torch.empty((total, ch, img_size, img_size), dtype=torch.uint8)
    lbls = torch.empty((total,), dtype=torch.long)

    for i, item in enumerate(ds):
        if i >= total:
            break
        imgs[i] = tf(item[img_key])
        lbls[i] = int(item[lbl_key])
        if i % 10000 == 0:
            print(f"[cache] decoded {i}/{total}", flush=True)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tmp = out_path + ".tmp"
    torch.save({"images": imgs, "labels": lbls, "img_size": img_size}, tmp)
    os.replace(tmp, out_path)  # atomic — concurrent readers never see a partial file
    print(f"[cache] saved {out_path}  images={tuple(imgs.shape)}", flush=True)
    return out_path


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--img_size", type=int, default=64)
    ap.add_argument("--n_shards", type=int, default=8)
    ap.add_argument("--n_images", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    build_cache(a.dataset, a.img_size, a.n_shards, a.out, a.n_images)
