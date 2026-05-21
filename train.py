"""
Nano-Diffusion — flow matching on MNIST / CIFAR-10.

Usage:
    python train.py configs/mnist.yaml
    python train.py configs/cifar10.yaml hidden_dim=512 depth=12
"""
import copy
import math
import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import wandb

from data import ImageDataset
from model import DiT
from sample import log_samples
from eval import compute_fid


# ---------------------------------------------------------------------------
# Flow matching
# ---------------------------------------------------------------------------

def get_zt(x0, t):
    eps = torch.randn_like(x0)
    t4  = t.view(-1, 1, 1, 1)
    return t4 * x0 + (1 - t4) * eps   # t=0 → noise, t=1 → data


def sigmoid_weight(t):
    """Bell-shaped loss weight peaking at mid-SNR (equal signal and noise)."""
    t      = t.clamp(1e-5, 1 - 1e-5)
    log_snr = 2 * torch.log(t / (1 - t))
    s      = torch.sigmoid(log_snr)
    return s * (1 - s)


def compute_loss(model, x0, labels, cfg_dropout: float, texts=None):
    null = model.cls_embed.num_embeddings - 1
    drop = torch.rand(len(labels), device=labels.device) < cfg_dropout
    labels_in = labels.masked_fill(drop, null)

    # For semantic routing: drop texts to None on the same mask (unconditional)
    texts_in = None
    if texts is not None:
        texts_in = [("" if d.item() else t) for t, d in zip(texts, drop)]

    t       = torch.rand(len(x0), device=x0.device)
    zt      = get_zt(x0, t)
    x0_pred = model(zt, t, labels_in, texts=texts_in)

    w = sigmoid_weight(t).view(-1, 1, 1, 1)
    return (w * (x0_pred - x0).pow(2)).mean()


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

def make_ema(model):
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    return ema


@torch.no_grad()
def update_ema(ema, model, decay: float, step: int):
    # ramp up decay so early EMA isn't dominated by the zero-initialized model
    decay = min(decay, (1 + step) / (10 + step))
    for ep, mp in zip(ema.parameters(), model.parameters()):
        ep.data.mul_(decay).add_(mp.data, alpha=1 - decay)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to YAML config")
    args, overrides = parser.parse_known_args()

    cfg = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(overrides))
    print(OmegaConf.to_yaml(cfg))

    device = torch.device(cfg.device)
    Path("checkpoints").mkdir(exist_ok=True)

    # data
    dataset = ImageDataset(cfg.dataset, img_size=cfg.img_size)
    loader  = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=(cfg.device != "cpu"), drop_last=True,
    )

    # model + EMA
    use_routing = getattr(cfg, "use_semantic_routing", False)
    model = DiT(
        img_size=cfg.img_size, patch_size=cfg.patch_size,
        channels=cfg.channels, num_classes=cfg.num_classes,
        d=cfg.hidden_dim, depth=cfg.depth, heads=cfg.num_heads,
        use_semantic_routing=use_routing,
        llm_model_name=getattr(cfg, "llm_model_name", None),
    ).to(device)
    ema = make_ema(model)
    print(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    if device.type == "cuda":
        model = torch.compile(model)

    # optimizer + cosine LR with linear warmup
    opt         = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.0)
    total_steps = cfg.epochs * len(loader)
    warmup      = int(0.05 * total_steps)

    def lr_lambda(step):
        if step < warmup:
            return step / max(warmup, 1)
        p = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * p))

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    os.environ.setdefault("WANDB_SILENT", "true")
    run  = wandb.init(project=cfg.wandb_project, name=cfg.run_name,
                      config=OmegaConf.to_container(cfg, resolve=True))
    step = 0

    # class index → text description lookup (only used when routing is on)
    class_texts = None
    if use_routing:
        from routing import CLASS_TEXTS
        class_texts = CLASS_TEXTS[cfg.dataset]

    for epoch in range(cfg.epochs):
        model.train()
        for x, labels in loader:
            x, labels = x.to(device), labels.to(device)

            texts = [class_texts[i] for i in labels.tolist()] if class_texts else None
            loss = compute_loss(model, x, labels, cfg.cfg_dropout, texts=texts)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            scheduler.step()
            update_ema(ema, model, cfg.ema_decay, step)
            step += 1

            run.log({"loss": loss.item(), "lr": scheduler.get_last_lr()[0]}, step=step)

            if step % cfg.eval_interval == 0:
                log_samples(ema, cfg, device, step, run)
                torch.save(
                    {"model": model.state_dict(), "ema": ema.state_dict(),
                     "opt": opt.state_dict(), "step": step,
                     "cfg": OmegaConf.to_container(cfg, resolve=True)},
                    f"checkpoints/{cfg.run_name}_step{step:07d}.pt",
                )

        print(f"epoch {epoch + 1}/{cfg.epochs}  step {step}  loss {loss.item():.4f}")

        if (epoch + 1) % cfg.fid_every_n_epochs == 0:
            fid = compute_fid(ema, cfg, device, n_samples=cfg.fid_samples)
            run.log({"fid": fid}, step=step)
            print(f"  FID: {fid:.2f}")

    run.finish()


if __name__ == "__main__":
    main()
