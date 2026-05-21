"""
Sampling utilities used by train.py for validation and as a standalone script.

Usage (standalone):
    python sample.py configs/mnist.yaml checkpoints/run_step5000.pt
    python sample.py configs/cifar10.yaml checkpoints/run_step5000.pt --steps 100 --cfg_scale 4.0 --out grid.png
"""
import argparse

import torch
import torchvision
import wandb
from omegaconf import OmegaConf

from model import DiT


@torch.no_grad()
def sample(model, labels, n_steps: int, cfg_scale: float, device):
    """Euler ODE with classifier-free guidance. Returns images in [-1, 1]."""
    B  = len(labels)
    z  = torch.randn(B, model.channels, model.img_size, model.img_size, device=device)
    labels = labels.to(device)
    null   = torch.full_like(labels, model.cls_embed.num_embeddings - 1)
    dt     = 1.0 / n_steps

    for i in range(n_steps):
        t = torch.full((B,), i / n_steps, device=device)
        x0_cond = model(z, t, labels)
        if cfg_scale != 1.0:
            x0_uncond = model(z, t, null)
            x0 = x0_uncond + cfg_scale * (x0_cond - x0_uncond)
        else:
            x0 = x0_cond
        if i < n_steps - 1:
            # velocity: v = (x0 - z) / (1 - t), Euler step
            v = (x0 - z) / (1.0 - i / n_steps + 1e-8)
            z = z + dt * v
        else:
            z = x0

    return z.clamp(-1.0, 1.0)


def log_samples(model, cfg, device, step, run):
    """Generate a class-grid and log to wandb. Called from train.py every eval_interval steps."""
    was_training = model.training
    model.eval()
    labels = torch.arange(cfg.num_classes, device=device).repeat(cfg.n_per_class)
    images = sample(model, labels, cfg.n_sample_steps, cfg.cfg_scale, device)
    images = (images + 1) / 2  # [-1, 1] -> [0, 1]
    grid   = torchvision.utils.make_grid(images, nrow=cfg.num_classes, pad_value=1.0)
    run.log({"samples": wandb.Image(grid)}, step=step)
    if was_training:
        model.train()


# ---------------------------------------------------------------------------
# Standalone script
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config",      help="Path to YAML config")
    parser.add_argument("checkpoint",  help="Path to .pt checkpoint")
    parser.add_argument("--out",       default="samples.png")
    parser.add_argument("--steps",     type=int,   default=None)
    parser.add_argument("--cfg_scale", type=float, default=None)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.steps     is not None: cfg.n_sample_steps = args.steps
    if args.cfg_scale is not None: cfg.cfg_scale      = args.cfg_scale

    device = torch.device(cfg.device)
    model  = DiT(
        img_size=cfg.img_size, patch_size=cfg.patch_size,
        channels=cfg.channels, num_classes=cfg.num_classes,
        d=cfg.hidden_dim, depth=cfg.depth, heads=cfg.num_heads,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["ema"])
    model.eval()

    labels = torch.arange(cfg.num_classes, device=device).repeat(cfg.n_per_class)
    images = sample(model, labels, cfg.n_sample_steps, cfg.cfg_scale, device)
    images = (images + 1) / 2
    grid   = torchvision.utils.make_grid(images, nrow=cfg.num_classes, pad_value=1.0)
    torchvision.utils.save_image(grid, args.out)
    print(f"saved → {args.out}")
