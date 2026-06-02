"""
Tokenizer reconstruction sanity check — no diffusion model involved.

Encodes real images through the Cosmos tokenizer and decodes them straight
back, to verify the tokenizer integration is faithful before trusting any FID.
For discrete (DI) tokenizers this also reports the FSQ round-trip index
accuracy, which isolates the decode-path FSQ math (`_codes_to_indices`) from
model quality.

    python recon.py configs/imagenet256_cosmos_di.yaml --n 16
"""
import argparse

import torch
import torchvision
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from data import StreamingImageDataset
from tokenizer import build_tokenizer


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--out", default="checkpoints/recon.png")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    tok = build_tokenizer(cfg, device)
    assert tok is not None, "config has use_tokenizer=false — nothing to reconstruct"

    # stream just enough images; no full-split download
    ds = StreamingImageDataset(cfg.dataset, img_size=cfg.img_size, shuffle_buffer=0)
    imgs, _ = next(iter(DataLoader(ds, batch_size=args.n, num_workers=0)))
    imgs = imgs.to(device)                      # [-1, 1], (n, 3, H, W)

    z = tok.encode(imgs)
    recon = tok.decode(z)                       # [-1, 1]

    mse = (recon - imgs).pow(2).mean().item()
    # pixel range is 2 ([-1, 1]) → peak signal² = 4
    psnr = 10 * torch.log10(torch.tensor(4.0 / max(mse, 1e-12))).item()
    print(f"tokenizer: {cfg.tokenizer_name}  ({'discrete' if tok.is_discrete else 'continuous'})")
    print(f"latent shape: {tuple(z.shape)}")
    print(f"reconstruction MSE: {mse:.5f}   PSNR: {psnr:.2f} dB")
    if tok.is_discrete:
        acc = tok.roundtrip_accuracy(imgs)
        print(f"FSQ round-trip index accuracy: {acc * 100:.2f}%  "
              f"(≈100% ⇒ decode FSQ math matches this build)")

    # top row: inputs, bottom row: reconstructions
    panel = torch.cat([(imgs + 1) / 2, (recon + 1) / 2], dim=0).clamp(0, 1)
    grid = torchvision.utils.make_grid(panel, nrow=args.n, pad_value=1.0)
    torchvision.utils.save_image(grid, args.out)
    print(f"saved comparison → {args.out}  (top: input, bottom: reconstruction)")


if __name__ == "__main__":
    main()
