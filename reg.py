"""
REG — Representation Entanglement for Generation  (arxiv 2507.01467)

A frozen DINOv2-B encodes each clean image to a single [CLS] token.
During training, that token is noised with the same flow schedule as the
image patches and prepended to the patch sequence.  The model predicts both
the image and the semantic token jointly; a small auxiliary loss (β) on the
semantic prediction substantially accelerates convergence.

During inference, cls_T is initialised from N(0, I) and Euler-stepped
alongside the image, so semantic guidance is active throughout denoising.

Usage (from train.py / sample.py):
    encoder  = REGEncoder("facebook/dinov2-base")
    cls_0    = encoder.encode(x0_images)          # (B, 768)
    cls_t, _ = noise_sem_token(cls_0, t)          # (B, 768)
    # pass cls_t as sem_token= to DiT.forward
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class REGEncoder(nn.Module):
    """Frozen DINOv2 encoder that returns the [CLS] token for each image."""

    def __init__(self, model_name: str = "facebook/dinov2-base"):
        super().__init__()
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(model_name)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.hidden_size = self.model.config.hidden_size

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, C, H, W) float32 in [-1, 1]
        Returns (B, hidden_size) — the DINOv2 [CLS] token for each image.
        """
        print(x.max(), x.min())
        x = (x + 1.0) / 2.0  # [-1,1] → [0,1]
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        if x.shape[1] == 1:  # grayscale → RGB
            x = x.repeat(1, 3, 1, 1)
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        outputs = self.model(pixel_values=(x - mean) / std)
        return outputs.last_hidden_state[:, 0]  # (B, hidden_size)


def noise_sem_token(
    cls_0: torch.Tensor,
    t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Flow-matching noise for the semantic CLS token — mirrors get_zt in train.py:
      cls_t = t * cls_0 + (1 - t) * ε,   ε ~ N(0, I)

    Returns (cls_t, ε).
    """
    eps = torch.randn_like(cls_0)
    t_ = t.unsqueeze(-1)  # (B, 1) to broadcast over hidden_dim
    cls_t = t_ * cls_0 + (1.0 - t_) * eps
    return cls_t, eps
