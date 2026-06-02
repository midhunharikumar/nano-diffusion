"""
Cosmos continuous-image tokenizer wrapper (NVIDIA/Cosmos-Tokenizer).

Lets the diffusion model train in a compressed latent space instead of raw
pixels.  The tokenizer is frozen — it only encodes images → latents (the DiT's
training target) and decodes latents → images (for sampling / FID).

    images [-1, 1], (B, 3, H, W)   --encode-->   latents (B, C, H/f, W/f)
    latents                        --decode-->   images  [-1, 1], (B, 3, H, W)

Install (the package is not on PyPI):

    pip install git+https://github.com/NVIDIA/Cosmos-Tokenizer.git

Checkpoints auto-download from HuggingFace on first use (the CI models are
public; HF_TOKEN is only needed if you hit a rate limit).

Latent normalisation
--------------------
Flow matching mixes the data tensor with N(0, I) noise, so the latents should
be roughly unit-variance.  Cosmos latents are not exactly standardised, so we
apply  z_norm = (z - latent_shift) * latent_scale  on encode and invert it on
decode.  Defaults are identity (scale=1, shift=0); estimate dataset-specific
values with:

    python tokenizer.py configs/imagenet256_cosmos.yaml
"""
import math
import os

import torch
import torch.nn as nn

# Continuous tokenizers — plain autoencoders, no codebook.
# (latent_channels, spatial_compression).
COSMOS_CONTINUOUS_MODELS = {
    "Cosmos-0.1-Tokenizer-CI8x8":   dict(latent_channels=4, compression=8),
    "Cosmos-0.1-Tokenizer-CI16x16": dict(latent_channels=4, compression=16),
}

# Discrete tokenizers — FSQ quantizer with a codebook. fsq_levels factorise the
# ~64K vocabulary (prod(levels)); len(fsq_levels) is the number of continuous
# pre-quantization channels the encoder exposes (the diffusion target).
# Levels [8,8,8,5,5,5] = 64000 — the FSQ paper's config for a 64K codebook.
COSMOS_DISCRETE_MODELS = {
    "Cosmos-0.1-Tokenizer-DI8x8":   dict(fsq_levels=[8, 8, 8, 5, 5, 5], compression=8),
    "Cosmos-0.1-Tokenizer-DI16x16": dict(fsq_levels=[8, 8, 8, 5, 5, 5], compression=16),
}


def is_discrete(tokenizer_name: str) -> bool:
    return tokenizer_name in COSMOS_DISCRETE_MODELS


def fsq_levels(tokenizer_name: str) -> list[int] | None:
    """Per-dimension FSQ levels for a discrete tokenizer, else None."""
    spec = COSMOS_DISCRETE_MODELS.get(tokenizer_name)
    return list(spec["fsq_levels"]) if spec else None


def _spec(tokenizer_name: str) -> tuple[int, int]:
    """(latent_channels, compression) for any supported tokenizer."""
    if tokenizer_name in COSMOS_CONTINUOUS_MODELS:
        s = COSMOS_CONTINUOUS_MODELS[tokenizer_name]
        return s["latent_channels"], s["compression"]
    if tokenizer_name in COSMOS_DISCRETE_MODELS:
        s = COSMOS_DISCRETE_MODELS[tokenizer_name]
        return len(s["fsq_levels"]), s["compression"]
    raise ValueError(
        f"unknown tokenizer {tokenizer_name!r}; supported: "
        f"{list(COSMOS_CONTINUOUS_MODELS) + list(COSMOS_DISCRETE_MODELS)}"
    )


def latent_dims(tokenizer_name: str, img_size: int) -> tuple[int, int]:
    """Return (channels, spatial_size) of the latent grid for a pixel img_size."""
    channels, f = _spec(tokenizer_name)
    if img_size % f != 0:
        raise ValueError(
            f"img_size {img_size} not divisible by compression {f} for {tokenizer_name}"
        )
    return channels, img_size // f


def model_dims(cfg) -> tuple[int, int]:
    """(channels, img_size) the DiT should be built with.

    Latent dims when a tokenizer is enabled, otherwise the raw pixel dims.
    Use this everywhere a DiT is constructed so train / eval / sample agree.
    """
    if getattr(cfg, "use_tokenizer", False):
        return latent_dims(cfg.tokenizer_name, cfg.img_size)
    return cfg.channels, cfg.img_size


class CosmosTokenizer(nn.Module):
    """Frozen Cosmos image encoder/decoder pair (continuous or discrete)."""

    def __init__(
        self,
        name: str,
        latent_scale: float = 1.0,
        latent_shift: float = 0.0,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        cache_dir: str | None = None,
    ):
        cache_dir = cache_dir or os.environ.get("COSMOS_CACHE_DIR", "pretrained_ckpts")
        super().__init__()
        self.latent_channels, self.compression = _spec(name)  # also validates name
        self.name         = name
        self.dtype        = dtype
        self.device       = device
        self.latent_scale = latent_scale
        self.latent_shift = latent_shift
        self.is_discrete  = is_discrete(name)
        self.fsq_levels   = fsq_levels(name)            # None for continuous
        self.codebook_size = (math.prod(self.fsq_levels)
                              if self.fsq_levels else None)

        local_dir = self._download(name, cache_dir)
        try:
            from cosmos_tokenizer.image_lib import ImageTokenizer
        except ImportError as e:
            raise ImportError(
                "cosmos_tokenizer not installed. Install with:\n"
                "    pip install git+https://github.com/NVIDIA/Cosmos-Tokenizer.git"
            ) from e

        # ImageTokenizer wraps TorchScript .jit modules; load encoder/decoder
        # separately so we never pay for the half we don't need.
        self.encoder = ImageTokenizer(
            checkpoint_enc=os.path.join(local_dir, "encoder.jit"), device=device,
        )
        self.decoder = ImageTokenizer(
            checkpoint_dec=os.path.join(local_dir, "decoder.jit"), device=device,
        )

    @staticmethod
    def _download(name: str, cache_dir: str) -> str:
        from huggingface_hub import snapshot_download

        local_dir = os.path.join(cache_dir, name)
        # Idempotent: skips files already present in the cache dir.
        snapshot_download(
            repo_id=f"nvidia/{name}",
            local_dir=local_dir,
            allow_patterns=["*.jit"],
        )
        return local_dir

    def _raw_encode(self, images: torch.Tensor):
        """Run the encoder and return (continuous_latent[float32], indices_or_None).

        Continuous models return just the latent; discrete (FSQ) models return
        (indices, codes) — we diffuse on the continuous pre-quant `codes` and
        keep the integer `indices` for the cross-entropy target.
        """
        out = self.encoder.encode(images.to(self.device, self.dtype))
        if self.is_discrete:
            indices, codes = out
            z = codes.float()
            if z.dim() == 5:          # (B, C, T=1, h, w) → (B, C, h, w)
                z = z.squeeze(2)
            return z, indices
        (z,) = out
        return z.float(), None

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """images in [-1, 1] → normalised latents (float32)."""
        z, _ = self._raw_encode(images)
        return (z - self.latent_shift) * self.latent_scale

    @torch.no_grad()
    def encode_with_targets(self, images: torch.Tensor):
        """images → (normalised latents, per-dim FSQ level targets).

        targets is (B, n_dims, h, w) long, each entry the level index in
        [0, fsq_levels[d]) — the factorised cross-entropy classification target.
        Discrete tokenizers only.
        """
        if not self.is_discrete:
            raise RuntimeError(
                f"{self.name} is continuous and has no codebook; "
                "use a discrete DI tokenizer for codebook cross-entropy."
            )
        z, indices = self._raw_encode(images)
        latents = (z - self.latent_shift) * self.latent_scale
        targets = self._decompose_indices(indices)
        return latents, targets

    def _decompose_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Flat FSQ index → per-dimension level indices via mixed-radix decode.

        Cosmos FSQuantizer.codes_to_indices is 0-based with
        basis = cumprod([1] + levels[:-1]) (first dim least-significant), so
        index = sum_d level_d * basis_d and level_d = (index // basis_d) % L_d.
        """
        idx = indices.long()
        if idx.dim() == 4 and idx.shape[1] == 1:   # (B, 1, h, w) → (B, h, w)
            idx = idx[:, 0]
        basis = 1
        per_dim = []
        for L in self.fsq_levels:
            per_dim.append((idx // basis) % L)
            basis *= L
        return torch.stack(per_dim, dim=1)         # (B, n_dims, h, w)

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """normalised latents → images clamped to [-1, 1] (float32).

        For discrete tokenizers the continuous prediction is FSQ-quantized back
        to integer indices (the decoder's expected input) before decoding.
        """
        z = z.to(self.device).float() / self.latent_scale + self.latent_shift
        if self.is_discrete:
            indices = self._codes_to_indices(z)
            img = self.decoder.decode(indices)
        else:
            img = self.decoder.decode(z.to(self.dtype))
        return img.float().clamp(-1.0, 1.0)

    # --- FSQ index encoding — used only for the decode path ---
    def _fsq_levels_tensor(self, z):
        return torch.tensor(self.fsq_levels, device=z.device, dtype=z.dtype).view(1, -1, 1, 1)

    def _codes_to_indices(self, z: torch.Tensor) -> torch.Tensor:
        """Normalised-quantized codes (B, n_dims, h, w) → flat 0-based FSQ indices (B, h, w).

        Cosmos `codes` live in the normalised quantized space (quantized /
        half_width), so we invert _scale_and_shift (z*half_width + half_width)
        to recover the per-dim level, snapping any off-grid diffusion prediction
        with round+clamp. Matches FSQuantizer.codes_to_indices:
        index = sum_d level_d * basis_d, 0-based, basis = cumprod([1]+levels[:-1]).
        """
        levels = self._fsq_levels_tensor(z)
        half_width = levels // 2
        level = (z * half_width + half_width).round().clamp_(min=0)
        level = torch.minimum(level, levels - 1).long()                  # (B, n_dims, h, w)
        idx = torch.zeros(z.shape[0], *z.shape[-2:], device=z.device, dtype=torch.long)
        basis = 1
        for d, L in enumerate(self.fsq_levels):
            idx = idx + level[:, d] * basis
            basis *= L
        return idx

    @torch.no_grad()
    def roundtrip_accuracy(self, images: torch.Tensor) -> float:
        """Fraction of FSQ indices our quantizer reproduces vs the encoder's own
        indices.  ~1.0 confirms the decode-path FSQ math matches this build."""
        z, indices = self._raw_encode(images)
        ours = self._codes_to_indices(z)
        ref = indices.long()
        if ref.dim() == 4 and ref.shape[1] == 1:
            ref = ref[:, 0]
        return (ours == ref).float().mean().item()


def build_tokenizer(cfg, device) -> "CosmosTokenizer | None":
    """Construct the tokenizer from a config, or None when disabled."""
    if not getattr(cfg, "use_tokenizer", False):
        return None
    return CosmosTokenizer(
        cfg.tokenizer_name,
        latent_scale=getattr(cfg, "latent_scale", 1.0),
        latent_shift=getattr(cfg, "latent_shift", 0.0),
        device=str(device),
    )


# ---------------------------------------------------------------------------
# Latent-stats estimator — run once per dataset to pick latent_scale.
# Reports per-channel mean/std of the *un-normalised* latents; a good
# latent_scale is ~1/std so the encoded latents land near unit variance.
# ---------------------------------------------------------------------------
def _estimate_stats(cfg, device, n_batches: int = 20):
    from torch.utils.data import DataLoader

    from data import ImageDataset

    # raw stats — disable normalisation while measuring
    tok = CosmosTokenizer(cfg.tokenizer_name, latent_scale=1.0, latent_shift=0.0,
                          device=str(device))
    ds = ImageDataset(cfg.dataset, img_size=cfg.img_size)
    loader = DataLoader(ds, batch_size=getattr(cfg, "batch_size", 32),
                        shuffle=True, num_workers=2)

    sums = sqs = count = 0
    for i, (imgs, _) in enumerate(loader):
        if i >= n_batches:
            break
        z = tok.encode(imgs.to(device))          # (B, C, h, w)
        sums = sums + z.sum(dim=(0, 2, 3))
        sqs = sqs + z.pow(2).sum(dim=(0, 2, 3))
        count += z.numel() // z.shape[1]
    mean = sums / count
    std = (sqs / count - mean.pow(2)).clamp_min(0).sqrt()
    print(f"tokenizer: {cfg.tokenizer_name}  dataset: {cfg.dataset}  img_size: {cfg.img_size}")
    print(f"per-channel mean: {mean.tolist()}")
    print(f"per-channel std:  {std.tolist()}")
    print(f"suggested latent_shift: {mean.mean().item():.4f}")
    print(f"suggested latent_scale: {1.0 / std.mean().item():.4f}  (≈ 1/std)")


if __name__ == "__main__":
    import argparse

    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser(description="Estimate Cosmos latent statistics")
    parser.add_argument("config")
    parser.add_argument("--n_batches", type=int, default=20)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    _estimate_stats(cfg, device, n_batches=args.n_batches)
