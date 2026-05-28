"""Mixtrain model: load a trained DiT checkpoint and generate class-conditioned
samples via Euler ODE + classifier-free guidance.

Deployed via:
    python mixtrain_launcher.py deploy-model \\
        --checkpoint checkpoints/cifar10_step0010000_fid12.5.pt \\
        --name nano-cifar10

Then called via:
    python mixtrain_launcher.py sample --model nano-cifar10 --class-label 7
"""
from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

from mixtrain import MixModel, Sandbox
from mixtrain.types import Image

REPO = Path(__file__).resolve().parent


class DiffusionSampler(MixModel):
    """Load a flow-matching DiT checkpoint and serve image samples."""

    # Same base image as the training workflow so checkpoint torch/CUDA versions
    # line up. Deps come from requirements.txt installed in setup() below.
    _sandbox = Sandbox(
        image="pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime",
        gpu="L4",
        memory=16384,
        timeout=600,
        min_containers=0,
    )

    def setup(self, checkpoint_path: str = ""):
        """Install deps, then load checkpoint config + EMA weights.

        Args:
            checkpoint_path: Path to a .pt produced by train.py. If empty,
                falls back to `model_weights.pt` next to this file — which is
                where `mixtrain_launcher deploy-model` bundles the checkpoint.
        """
        import subprocess as _sp

        req = REPO / "requirements.txt"
        if req.exists():
            _sp.check_call([
                sys.executable, "-m", "pip", "install", "-q",
                "--upgrade", "--no-cache-dir",
                "-r", str(req),
            ])

        import torch
        from omegaconf import OmegaConf

        sys.path.insert(0, str(REPO))
        from model import DiT  # noqa: E402

        self.torch = torch
        if not checkpoint_path:
            checkpoint_path = str(REPO / "model_weights.pt")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.cfg = OmegaConf.create(ckpt["cfg"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = DiT(
            img_size=self.cfg.img_size,
            patch_size=self.cfg.patch_size,
            channels=self.cfg.channels,
            num_classes=self.cfg.num_classes,
            d=self.cfg.hidden_dim,
            depth=self.cfg.depth,
            heads=self.cfg.num_heads,
        ).to(self.device)

        # torch.compile prefixes keys with _orig_mod.
        state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["ema"].items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"load_state_dict missing={len(missing)} unexpected={len(unexpected)}")
        model.eval()
        self.model = model
        self.checkpoint_path = checkpoint_path

    def run(
        self,
        class_label: int = 0,
        n_images: int = 1,
        n_steps: int = 50,
        cfg_scale: float = 3.0,
    ) -> dict:
        """Generate n_images samples of the given class.

        Args:
            class_label: Integer class index (0..num_classes-1).
            n_images: Number of samples to return.
            n_steps: Euler integration steps.
            cfg_scale: Classifier-free guidance scale.
        """
        from PIL import Image as PILImage

        sys.path.insert(0, str(REPO))
        from sample import sample  # noqa: E402

        labels = self.torch.tensor([class_label] * n_images, device=self.device)
        imgs = sample(self.model, labels, n_steps, cfg_scale, self.device)
        imgs = ((imgs + 1) / 2 * 255).clamp(0, 255).byte().cpu()

        out = []
        for img in imgs:
            arr = img.permute(1, 2, 0).numpy()
            if arr.shape[-1] == 1:
                arr = arr.squeeze(-1)  # grayscale (MNIST)
            pil = PILImage.fromarray(arr)
            buf = io.BytesIO()
            pil.save(buf, format="PNG")
            data_uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
            h, w = arr.shape[:2]
            out.append(Image(url=data_uri, width=w, height=h))

        return {
            "images": out,
            "class_label": class_label,
            "n_steps": n_steps,
            "cfg_scale": cfg_scale,
        }
