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
from data import _CFGS
from sample import sample as diffusion_sample, sample_with_reg


@torch.no_grad()
def compute_fid(model, cfg, device, n_samples: int = 2048, tokenizer=None) -> float:
    """Generate n_samples images and compute FID against the test split.

    Real images are always compared in pixel space (cfg.img_size); when a
    tokenizer is set, generated latents are decoded to pixels before scoring.
    """
    eval_split = _CFGS.get(cfg.dataset, {}).get("eval_split", "test")
    test_ds = ImageDataset(cfg.dataset, split=eval_split, img_size=cfg.img_size)

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
    sampler = sample_with_reg if getattr(model, "use_reg", False) else diffusion_sample
    n_gen = 0
    while n_gen < n_samples:
        batch_labels = labels_all[n_gen : n_gen + 128]
        imgs = sampler(model, batch_labels, cfg.n_sample_steps, cfg.cfg_scale, device,
                       tokenizer=tokenizer)
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

    cfg     = OmegaConf.load(args.config)
    device  = torch.device(cfg.device)
    use_reg = getattr(cfg, "use_reg", False)

    from tokenizer import build_tokenizer, fsq_levels, model_dims
    tokenizer = build_tokenizer(cfg, device)
    model_channels, model_img_size = model_dims(cfg)
    use_codebook_ce = getattr(cfg, "use_codebook_ce", False)
    fsq = fsq_levels(cfg.tokenizer_name) if use_codebook_ce else None

    model   = DiT(
        img_size=model_img_size, patch_size=cfg.patch_size,
        channels=model_channels, num_classes=cfg.num_classes,
        d=cfg.hidden_dim, depth=cfg.depth, heads=cfg.num_heads,
        use_reg=use_reg,
        reg_model_name=getattr(cfg, "reg_model_name", "facebook/dinov2-base"),
        use_tread=getattr(cfg, "use_tread", False),
        tread_selection_rate=getattr(cfg, "tread_selection_rate", 0.5),
        tread_route_start=getattr(cfg, "tread_route_start", 2),
        tread_route_end=getattr(cfg, "tread_route_end", -1),
        use_maskgit=getattr(cfg, "use_maskgit", False),
        maskgit_ratio=getattr(cfg, "maskgit_ratio", 0.5),
        use_moe=getattr(cfg, "use_moe", False),
        moe_num_experts=getattr(cfg, "moe_num_experts", 8),
        moe_num_always_on=getattr(cfg, "moe_num_always_on", 1),
        moe_capacity_factor=getattr(cfg, "moe_capacity_factor", 1.25),
        moe_every_n=getattr(cfg, "moe_every_n", 1),
        use_codebook_ce=use_codebook_ce,
        fsq_levels=fsq,
        ce_output=getattr(cfg, "ce_output", False),
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["ema"])
    model.eval()

    fid = compute_fid(model, cfg, device, n_samples=args.n_samples, tokenizer=tokenizer)
    print(f"FID: {fid:.2f}")
