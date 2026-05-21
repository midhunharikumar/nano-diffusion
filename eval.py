"""
FID evaluation against the dataset test split.

Standalone:
    python eval.py configs/cifar10.yaml checkpoints/run_step0001000.pt
    python eval.py configs/mnist.yaml   checkpoints/run_step0001000.pt --n_samples 5000
"""
import argparse

import torch
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from omegaconf import OmegaConf

from data import ImageDataset
from model import DiT
from sample import sample as diffusion_sample


@torch.no_grad()
def compute_fid(model, cfg, device, n_samples: int = 2048) -> float:
    """Generate n_samples images and compute FID against the test split."""
    test_ds = ImageDataset(cfg.dataset, split="test", img_size=cfg.img_size)

    # Inception v3 has known issues on MPS; run FID stats on CUDA or CPU
    fid_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(fid_device)

    # --- real images ---
    loader = DataLoader(test_ds, batch_size=128, shuffle=True, num_workers=0)
    n_real = 0
    for imgs, _ in loader:
        if n_real >= n_samples:
            break
        imgs = imgs[: n_samples - n_real]
        imgs = ((imgs + 1) / 2).clamp(0, 1)
        if imgs.shape[1] == 1:
            imgs = imgs.repeat(1, 3, 1, 1)   # grayscale → RGB for inception
        fid.update(imgs.to(fid_device), real=True)
        n_real += len(imgs)

    # --- generated images ---
    was_training = model.training
    model.eval()
    labels_all = (
        torch.arange(cfg.num_classes)
        .repeat((n_samples + cfg.num_classes - 1) // cfg.num_classes)[:n_samples]
    )
    n_gen = 0
    while n_gen < n_samples:
        batch_labels = labels_all[n_gen : n_gen + 128]
        imgs = diffusion_sample(model, batch_labels, cfg.n_sample_steps, cfg.cfg_scale, device)
        imgs = ((imgs + 1) / 2).clamp(0, 1).cpu()
        if imgs.shape[1] == 1:
            imgs = imgs.repeat(1, 3, 1, 1)
        fid.update(imgs.to(fid_device), real=False)
        n_gen += len(batch_labels)
    if was_training:
        model.train()

    score = fid.compute().item()
    del fid   # free inception weights from memory
    return score


# ---------------------------------------------------------------------------
# Standalone
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("--n_samples", type=int, default=2048)
    args = parser.parse_args()

    cfg    = OmegaConf.load(args.config)
    device = torch.device(cfg.device)
    model  = DiT(
        img_size=cfg.img_size, patch_size=cfg.patch_size,
        channels=cfg.channels, num_classes=cfg.num_classes,
        d=cfg.hidden_dim, depth=cfg.depth, heads=cfg.num_heads,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["ema"])
    model.eval()

    fid = compute_fid(model, cfg, device, n_samples=args.n_samples)
    print(f"FID: {fid:.2f}")
