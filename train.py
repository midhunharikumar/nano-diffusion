"""
Nano-Diffusion — flow matching on MNIST / CIFAR-10.

Usage:
    python train.py configs/mnist.yaml
    python train.py configs/cifar10.yaml hidden_dim=512 depth=12
"""

import argparse
import copy
import math
import os
import time
import uuid
from pathlib import Path

import torch
import torch.nn.functional as F
from einops import rearrange
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

import wandb
from tqdm import tqdm
from data import _CFGS, ImageDataset, StreamingImageDataset
from eval import compute_fid
from gcs import GCSCheckpointUploader
from model import DiT
from sample import log_samples

# ---------------------------------------------------------------------------
# Flow matching
# ---------------------------------------------------------------------------


def get_zt(x0, t):
    eps = torch.randn_like(x0)
    t4 = t.view(-1, 1, 1, 1)
    return t4 * x0 + (1 - t4) * eps  # t=0 → noise, t=1 → data


def sigmoid_weight(t):
    """Bell-shaped loss weight peaking at mid-SNR (equal signal and noise)."""
    t = t.clamp(1e-5, 1 - 1e-5)
    log_snr = 2 * torch.log(t / (1 - t))
    s = torch.sigmoid(log_snr)
    return s * (1 - s)


def sample_timesteps(
    n: int, img_size: int, base_size: int = 32, shifted: bool = False, device="cpu"
) -> torch.Tensor:
    """
    Sample flow-matching timesteps in [0, 1].

    shifted=False : uniform (original behaviour)
    shifted=True  : logit-normal shifted by resolution (simple-diffusion §3.4,
                    arxiv 2301.11093).  Larger images bias toward lower t (higher
                    noise) so the model spends more training time on global structure.

      t = sigmoid(u - shift),  u ~ N(0,1),  shift = ln(img_size / base_size)

    At base_size the shift is 0 (standard logit-normal).
    At 256px (base 32px) shift ≈ 2.08 → distribution moves toward t≈0.
    """
    if not shifted:
        return torch.rand(n, device=device)
    shift = math.log(img_size / base_size)
    u = torch.randn(n, device=device)
    return torch.sigmoid(u - shift)


def compute_loss(
    model,
    x0,
    labels,
    cfg_dropout: float,
    texts=None,
    use_reg: bool = False,
    reg_beta: float = 0.03,
    shifted_t: bool = False,
):
    null = model.cls_embed.num_embeddings - 1
    drop = torch.rand(len(labels), device=labels.device) < cfg_dropout
    labels_in = labels.masked_fill(drop, null)

    # For semantic routing: drop texts to None on the same mask (unconditional)
    texts_in = None
    if texts is not None:
        texts_in = [("" if d.item() else t) for t, d in zip(texts, drop)]

    t = sample_timesteps(
        len(x0), img_size=x0.shape[-1], shifted=shifted_t, device=x0.device
    )
    zt = get_zt(x0, t)

    # REG: extract the DINOv2 CLS token from the clean image and noise it
    sem_token = None
    cls_0 = None
    if use_reg and getattr(model, "reg_encoder", None) is not None:
        from reg import noise_sem_token

        cls_0 = model.reg_encoder.encode(x0)
        cls_0 = F.normalize(cls_0, dim=-1)        # unit sphere — raw DINOv2 norms (~50–100)
                                                   # cause trivial MSE minimisation otherwise
        sem_token, _ = noise_sem_token(cls_0, t)  # (B, reg_hidden_size)

        # CFG consistency: at inference the unconditional pass has no sem_token (pure noise).
        # Match that here: for CFG-dropped samples replace sem_token with N(0,I) so the
        # model never sees cls_0 signal on unconditional examples.
        if drop.any():
            sem_token = sem_token.clone()
            sem_token[drop] = torch.randn_like(sem_token[drop])

    result = model(zt, t, labels_in, texts=texts_in, sem_token=sem_token)
    x0_pred, cls0_pred, ids_keep = result

    if ids_keep is not None:
        # MaskGIT active: x0_pred is (B, n_keep, patch_dim); compute loss in patch space
        raw = getattr(model, "_orig_mod", model)
        p = raw.patch_size
        x0_patches = rearrange(x0, "b c (h p1) (w p2) -> b (h w) (c p1 p2)", p1=p, p2=p)
        patch_dim = x0_patches.size(-1)
        x0_kept = x0_patches.gather(1, ids_keep.unsqueeze(-1).expand(-1, -1, patch_dim))
        w = sigmoid_weight(t).view(-1, 1, 1)
        img_loss = (w * (x0_pred - x0_kept).pow(2)).mean()
    else:
        w = sigmoid_weight(t).view(-1, 1, 1, 1)
        img_loss = (w * (x0_pred - x0).pow(2)).mean()
    aux_loss = None

    # Auxiliary loss: predict the clean DINOv2 CLS token.
    # No per-timestep weighting here — the paper integrates uniformly over t.
    # Applying sigmoid_weight would collapse the loss to ~0 at the low-t values
    # that shifted timestep sampling draws most often.
    if sem_token is not None:
        aux_loss = reg_beta * (cls0_pred - cls_0).pow(2).mean()

    total_loss = img_loss + aux_loss if aux_loss is not None else img_loss
    return total_loss, img_loss, aux_loss


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------


def make_ema(model):
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    # Share the frozen REG encoder between model and EMA to avoid doubling memory
    if getattr(model, "reg_encoder", None) is not None:
        ema.reg_encoder = model.reg_encoder
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
    parser.add_argument(
        "--run_name",
        default=None,
        help="Override run name (UUID suffix always appended)",
    )
    parser.add_argument(
        "--max_runtime",
        type=int,
        default=None,
        metavar="SECONDS",
        help="Stop training after this many seconds, save checkpoint, and exit",
    )
    parser.add_argument(
        "--gcp_bucket",
        default="",
        help="GCS bucket name for checkpoint uploads (e.g. 'checkpoints')",
    )
    args, overrides = parser.parse_known_args()

    cfg = OmegaConf.merge(
        OmegaConf.load(args.config), OmegaConf.from_dotlist(overrides)
    )

    base_name = args.run_name or cfg.run_name
    run_name = f"{base_name}_{uuid.uuid4().hex[:8]}"
    print(OmegaConf.to_yaml(cfg))
    print(f"run: {run_name}\n")

    gcs_bucket = getattr(cfg, "gcp_bucket", "") or args.gcp_bucket
    gcs = GCSCheckpointUploader(gcs_bucket, run_name) if gcs_bucket else None

    # TF32 — free ~10% speedup on Ampere+ (A100, RTX 30xx+); no-op on other hardware
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device(cfg.device)
    Path("checkpoints").mkdir(exist_ok=True)

    # data — streaming skips the full download; map-style caches locally
    streaming = getattr(cfg, "streaming", False)
    nw = cfg.num_workers
    if streaming:
        dataset = StreamingImageDataset(cfg.dataset, img_size=cfg.img_size)
        loader = DataLoader(
            dataset, batch_size=cfg.batch_size, shuffle=False,
            num_workers=nw, pin_memory=False, drop_last=True,
            persistent_workers=False,   # HF streaming workers can't be safely persisted;
            prefetch_factor=2 if nw > 0 else None,  # keep next batch ready
        )
        ds_size = _CFGS[cfg.dataset].get("size")
        epoch_len = getattr(
            cfg, "steps_per_epoch", ds_size // cfg.batch_size if ds_size else None
        )
        assert epoch_len, (
            f"streaming=true requires steps_per_epoch in config or a known dataset size "
            f"(add size to _CFGS['{cfg.dataset}'])"
        )
    else:
        dataset = ImageDataset(cfg.dataset, img_size=cfg.img_size)
        loader = DataLoader(
            dataset, batch_size=cfg.batch_size, shuffle=True,
            num_workers=nw, pin_memory=(cfg.device != "cpu"), drop_last=True,
            persistent_workers=(nw > 0),
        )
        epoch_len = len(loader)

    # model + EMA
    use_routing = getattr(cfg, "use_semantic_routing", False)
    use_reg = getattr(cfg, "use_reg", False)
    model = DiT(
        img_size=cfg.img_size,
        patch_size=cfg.patch_size,
        channels=cfg.channels,
        num_classes=cfg.num_classes,
        d=cfg.hidden_dim,
        depth=cfg.depth,
        heads=cfg.num_heads,
        use_semantic_routing=use_routing,
        llm_model_name=getattr(cfg, "llm_model_name", None),
        checkpoint_every=getattr(cfg, "checkpoint_every", 0),
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
    ).to(device)
    ema = make_ema(model)
    print(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    if device.type == "cuda" and getattr(cfg, "compile", True):
        model = torch.compile(model)

    # optimizer + cosine LR with linear warmup
    # step counts optimizer updates; total_steps is independent of grad_accum
    grad_accum = getattr(cfg, "grad_accum_steps", 1)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=0.0,
        fused=(device.type == "cuda"),   # fused kernel on CUDA; silent no-op elsewhere
    )
    total_steps = cfg.epochs * epoch_len // grad_accum
    warmup = getattr(cfg, "warmup_steps", 500)  # fixed steps; override in config

    def lr_lambda(step):
        if step < warmup:
            return step / max(warmup, 1)
        p = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * p))

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    os.environ.setdefault("WANDB_SILENT", "true")
    run = wandb.init(
        project=cfg.wandb_project,
        name=run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    step = 0
    t_start = time.time()
    opt.zero_grad(set_to_none=True)

    # class index → text description lookup (only used when routing is on)
    class_texts = None
    if use_routing:
        from routing import CLASS_TEXTS

        class_texts = CLASS_TEXTS[cfg.dataset]
    print("Starting Training run !")
    epoch_bar = tqdm(range(cfg.epochs), desc="epochs", unit="epoch", position=0)
    for epoch in epoch_bar:
        model.train()
        # accumulators reset each optimizer step
        accum_loss = accum_img_loss = accum_aux_loss = 0.0
        accum_aux_active = False

        batch_bar = tqdm(
            loader, desc=f"epoch {epoch + 1:>{len(str(cfg.epochs))}}/{cfg.epochs}",
            unit="batch", total=epoch_len, position=1, leave=False,
        )
        for i, (x, labels) in enumerate(batch_bar):
            x, labels = x.to(device), labels.to(device)

            texts = [class_texts[i] for i in labels.tolist()] if class_texts else None
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=(device.type in ("cuda", "cpu")),
            ):
                loss, img_loss, aux_loss = compute_loss(
                    model,
                    x,
                    labels,
                    cfg.cfg_dropout,
                    texts=texts,
                    use_reg=use_reg,
                    reg_beta=getattr(cfg, "reg_beta", 0.03),
                    shifted_t=getattr(cfg, "shifted_t", False),
                )
            (loss / grad_accum).backward()

            accum_loss += loss.item()
            accum_img_loss += img_loss.item()
            if aux_loss is not None:
                accum_aux_loss += aux_loss.item()
                accum_aux_active = True

            # optimizer update at the end of each accumulation window
            end_of_window = (i + 1) % grad_accum == 0 or (i + 1) == epoch_len
            if not end_of_window:
                continue

            batches_in_window = (i % grad_accum) + 1
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.grad_clip
            )
            opt.step()
            opt.zero_grad(set_to_none=True)
            scheduler.step()
            update_ema(ema, model, cfg.ema_decay, step)
            step += 1

            # runtime limit — sample, compute FID, save checkpoint, and exit cleanly
            if args.max_runtime and (time.time() - t_start) >= args.max_runtime:
                print(f"  max_runtime {args.max_runtime}s reached at step {step} — sampling")
                log_samples(ema, cfg, device, step, run)
                print(f"  computing FID")
                fid = compute_fid(ema, cfg, device, n_samples=cfg.fid_samples)
                run.log({"fid": fid}, step=step)
                print(f"  FID: {fid:.2f}")
                ckpt_path = f"checkpoints/{run_name}_step{step:07d}_fid{fid:.2f}_timeout.pt"
                torch.save(
                    {
                        "model": model.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "step": step,
                        "fid": fid,
                        "cfg": OmegaConf.to_container(cfg, resolve=True),
                    },
                    ckpt_path,
                )
                print(f"  saved {ckpt_path}")
                if gcs:
                    gcs.upload(ckpt_path)
                run.finish()
                return

            log = {
                "loss": accum_loss / batches_in_window,
                "grad_norm": grad_norm.item(),
                "lr": scheduler.get_last_lr()[0],
                "epoch": epoch + 1,
            }
            if accum_aux_active:
                log["loss/img"] = accum_img_loss / batches_in_window
                log["loss/aux"] = accum_aux_loss / batches_in_window
            run.log(log, step=step)

            # update inner bar postfix after every optimizer step
            pf = {"loss": f"{log['loss']:.4f}", "gnorm": f"{grad_norm.item():.2f}",
                  "lr": f"{scheduler.get_last_lr()[0]:.1e}", "step": step}
            if accum_aux_active:
                pf["img"] = f"{log['loss/img']:.4f}"
                pf["aux"] = f"{log['loss/aux']:.4f}"
            batch_bar.set_postfix(pf)

            accum_loss = accum_img_loss = accum_aux_loss = 0.0
            accum_aux_active = False

            if step % cfg.eval_interval == 0:
                log_samples(ema, cfg, device, step, run)

        batch_bar.close()
        last_loss = log["loss"]
        epoch_bar.set_postfix({"loss": f"{last_loss:.4f}", "step": step})

        if (epoch + 1) % cfg.fid_every_n_epochs == 0:
            fid = compute_fid(ema, cfg, device, n_samples=cfg.fid_samples)
            run.log({"fid": fid}, step=step)
            print(f"  FID: {fid:.2f}")
            ckpt_path = f"checkpoints/{run_name}_step{step:07d}_fid{fid:.2f}.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "ema": ema.state_dict(),
                    "opt": opt.state_dict(),
                    "step": step,
                    "fid": fid,
                    "cfg": OmegaConf.to_container(cfg, resolve=True),
                },
                ckpt_path,
            )
            if gcs:
                gcs.upload(ckpt_path)

    # Final eval — always run at the end of training if the last epoch didn't
    # already trigger a scheduled FID (i.e. epochs % fid_every_n_epochs != 0)
    is_final_epoch = cfg.epochs % cfg.fid_every_n_epochs != 0
    if is_final_epoch:
        print("→ final eval")
        log_samples(ema, cfg, device, step, run)
        fid = compute_fid(ema, cfg, device, n_samples=cfg.fid_samples)
        run.log({"fid": fid}, step=step)
        print(f"  FID: {fid:.2f}")
        ckpt_path = f"checkpoints/{run_name}_step{step:07d}_fid{fid:.2f}_final.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "ema": ema.state_dict(),
                "opt": opt.state_dict(),
                "step": step,
                "fid": fid,
                "cfg": OmegaConf.to_container(cfg, resolve=True),
            },
            ckpt_path,
        )
        if gcs:
            gcs.upload(ckpt_path)

    run.finish()


if __name__ == "__main__":
    main()
