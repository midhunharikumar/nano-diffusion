"""
Sampling utilities used by train.py for validation and as a standalone script.

Usage (standalone):
    python sample.py configs/mnist.yaml checkpoints/run_step5000.pt
    python sample.py configs/cifar10.yaml checkpoints/run_step5000.pt --steps 100 --cfg_scale 4.0 --out grid.png

Semantic routing (model must be trained with use_semantic_routing: true):
    python sample.py configs/cifar10.yaml checkpoints/run.pt --text "a red fire truck"
    python sample.py configs/mnist.yaml   checkpoints/run.pt --text "the number after six"
"""
import argparse

import torch
import torchvision
import wandb
from omegaconf import OmegaConf

from model import DiT


def _to_images(z, tokenizer):
    """Map the model's output (latents when a tokenizer is set, else pixels) to
    images in [-1, 1]."""
    if tokenizer is not None:
        return tokenizer.decode(z)
    return z.clamp(-1.0, 1.0)


@torch.no_grad()
def sample(model, labels, n_steps: int, cfg_scale: float, device, tokenizer=None):
    """Euler ODE with classifier-free guidance. Returns images in [-1, 1].

    When ``tokenizer`` is set the ODE runs in latent space and the final latent
    is decoded back to pixels.
    """
    B  = len(labels)
    z  = torch.randn(B, model.channels, model.img_size, model.img_size, device=device)
    labels = labels.to(device)
    null   = torch.full_like(labels, model.cls_embed.num_embeddings - 1)
    dt     = 1.0 / n_steps

    for i in range(n_steps):
        t = torch.full((B,), i / n_steps, device=device)
        x0_cond, _, _, _ = model(z, t, labels)
        if cfg_scale != 1.0:
            x0_uncond, _, _, _ = model(z, t, null)
            x0 = x0_uncond + cfg_scale * (x0_cond - x0_uncond)
        else:
            x0 = x0_cond
        if i < n_steps - 1:
            # velocity: v = (x0 - z) / (1 - t), Euler step
            v = (x0 - z) / (1.0 - i / n_steps + 1e-8)
            z = z + dt * v
        else:
            z = x0

    return _to_images(z, tokenizer)


def log_samples(model, cfg, device, step, run, tokenizer=None):
    """Generate a class-grid and log to wandb. Called from train.py every eval_interval steps."""
    was_training = model.training
    model.eval()
    # n_log_classes caps the grid width for datasets with many classes (e.g. ImageNet)
    n_show = min(cfg.num_classes, getattr(cfg, "n_log_classes", cfg.num_classes))
    labels = torch.arange(n_show, device=device).repeat(cfg.n_per_class)
    if getattr(model, "use_reg", False):
        images = sample_with_reg(model, labels, cfg.n_sample_steps, cfg.cfg_scale, device)
    else:
        images = sample(model, labels, cfg.n_sample_steps, cfg.cfg_scale, device,
                        tokenizer=tokenizer)
    images = (images + 1) / 2  # [-1, 1] -> [0, 1]
    grid   = torchvision.utils.make_grid(images, nrow=n_show, pad_value=1.0)
    run.log({"samples": wandb.Image(grid)}, step=step)
    if was_training:
        model.train()


# ---------------------------------------------------------------------------
# REG sampling — jointly denoises image patches and semantic CLS token
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_with_reg(model, labels, n_steps: int, cfg_scale: float, device,
                    tokenizer=None) -> torch.Tensor:
    """
    Euler ODE with REG (arxiv 2507.01467).  The DINOv2 CLS token is initialised
    from N(0, I) and denoised alongside image patches at every step, providing
    active semantic guidance throughout.  Model must be trained with use_reg=True.
    """
    B        = len(labels)
    z        = torch.randn(B, model.channels, model.img_size, model.img_size, device=device)
    cls_t    = torch.randn(B, model.reg_encoder.hidden_size, device=device)
    labels   = labels.to(device)
    null     = torch.full_like(labels, model.cls_embed.num_embeddings - 1)
    dt       = 1.0 / n_steps

    for i in range(n_steps):
        t = torch.full((B,), i / n_steps, device=device)

        x0_cond,  cls0_cond,  _, _ = model(z, t, labels, sem_token=cls_t)
        if cfg_scale != 1.0:
            x0_uncond, cls0_uncond, _, _ = model(z, t, null, sem_token=cls_t)
            x0   = x0_uncond  + cfg_scale * (x0_cond  - x0_uncond)
            cls0 = cls0_uncond + cfg_scale * (cls0_cond - cls0_uncond)
        else:
            x0   = x0_cond
            cls0 = cls0_cond

        if i < n_steps - 1:
            one_minus_t = 1.0 - i / n_steps + 1e-8
            v_img = (x0   - z)     / one_minus_t
            v_cls = (cls0 - cls_t) / one_minus_t
            z     = z     + dt * v_img
            cls_t = cls_t + dt * v_cls
        else:
            z = x0

    return _to_images(z, tokenizer)


# ---------------------------------------------------------------------------
# Semantic routing sampling
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_with_routing(model, texts: list[str], cfg, device, tokenizer=None) -> torch.Tensor:
    """Sample using semantic routing. Model must be trained with use_semantic_routing=True.
    texts: list of B free-text descriptions. Returns images in [-1, 1].
    """
    B   = len(texts)
    z   = torch.randn(B, model.channels, model.img_size, model.img_size, device=device)
    # null label = unconditional token (CFG)
    null_labels = torch.full((B,), model.cls_embed.num_embeddings - 1,
                              dtype=torch.long, device=device)
    empty_texts = [""] * B
    dt = 1.0 / cfg.n_sample_steps

    for i in range(cfg.n_sample_steps):
        t = torch.full((B,), i / cfg.n_sample_steps, device=device)
        x0_cond,   _, _, _ = model(z, t, null_labels, texts=texts)
        x0_uncond, _, _, _ = model(z, t, null_labels, texts=empty_texts)
        x0 = x0_uncond + cfg.cfg_scale * (x0_cond - x0_uncond)
        if i < cfg.n_sample_steps - 1:
            v = (x0 - z) / (1.0 - i / cfg.n_sample_steps + 1e-8)
            z = z + dt * v
        else:
            z = x0
    return _to_images(z, tokenizer)


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
    parser.add_argument("--text",      default=None,
                        help="Free-text prompt for semantic routing (model must support it)")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.steps     is not None: cfg.n_sample_steps = args.steps
    if args.cfg_scale is not None: cfg.cfg_scale      = args.cfg_scale

    use_routing = getattr(cfg, "use_semantic_routing", False)
    use_reg     = getattr(cfg, "use_reg", False)
    device = torch.device(cfg.device)

    from tokenizer import build_tokenizer, fsq_levels, model_dims
    tokenizer = build_tokenizer(cfg, device)
    model_channels, model_img_size = model_dims(cfg)
    use_codebook_ce = getattr(cfg, "use_codebook_ce", False)
    fsq = fsq_levels(cfg.tokenizer_name) if use_codebook_ce else None

    model  = DiT(
        img_size=model_img_size, patch_size=cfg.patch_size,
        channels=model_channels, num_classes=cfg.num_classes,
        d=cfg.hidden_dim, depth=cfg.depth, heads=cfg.num_heads,
        use_semantic_routing=use_routing,
        llm_model_name=getattr(cfg, "llm_model_name", None),
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

    if args.text and use_routing:
        images = sample_with_routing(model, [args.text], cfg, device, tokenizer=tokenizer)
        images = (images + 1) / 2
        torchvision.utils.save_image(images[0], args.out)
        print(f"saved → {args.out}  (text: '{args.text}')")
    elif use_reg:
        labels = torch.arange(cfg.num_classes, device=device).repeat(cfg.n_per_class)
        images = sample_with_reg(model, labels, cfg.n_sample_steps, cfg.cfg_scale, device,
                                 tokenizer=tokenizer)
        images = (images + 1) / 2
        grid   = torchvision.utils.make_grid(images, nrow=cfg.num_classes, pad_value=1.0)
        torchvision.utils.save_image(grid, args.out)
        print(f"saved → {args.out}  (REG)")
    else:
        labels = torch.arange(cfg.num_classes, device=device).repeat(cfg.n_per_class)
        images = sample(model, labels, cfg.n_sample_steps, cfg.cfg_scale, device,
                        tokenizer=tokenizer)
        images = (images + 1) / 2
        grid   = torchvision.utils.make_grid(images, nrow=cfg.num_classes, pad_value=1.0)
        torchvision.utils.save_image(grid, args.out)
        print(f"saved → {args.out}")
