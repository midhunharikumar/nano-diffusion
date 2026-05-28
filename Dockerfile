# syntax=docker/dockerfile:1.6
#
# Mixtrain sandbox image for nano-diffusion.
#
# Base: PyTorch 2.4.0 + CUDA 12.1 + cuDNN 9 + Python 3.11 (preinstalled).
# Built once per workflow/model deploy; picked up automatically by
# `mixtrain workflow create` / `mixtrain model create` when this file
# sits at the source root.
#
# To pin to a different CUDA/Torch combo, change BASE_IMAGE below.

ARG BASE_IMAGE=pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0" \
    WANDB_MODE=disabled

# System deps:
#   git           — HF Hub + transformers occasionally clone repos
#   git-lfs       — for large HF model weight pulls
#   ca-certificates, curl — TLS + downloads
#   build-essential — wheels that lack manylinux builds
#   libgl1, libglib2.0-0 — Pillow / opencv image I/O
#   ffmpeg        — torchvision video utilities (cheap insurance)
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        git-lfs \
        ca-certificates \
        curl \
        build-essential \
        libgl1 \
        libglib2.0-0 \
        ffmpeg \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install Python deps first so the layer caches across source edits.
# torch / torchvision are already in the base image; pip will respect the
# `>=2.1.0` constraint and skip reinstallation.
COPY requirements.txt /workspace/requirements.txt
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip install mixtrain

# Sanity-check the transformers stack at build time. torchmetrics eagerly
# imports its text submodule, which lazy-loads transformers.AutoModel; if any
# transitive dep (tokenizers, safetensors, accelerate) is missing or
# version-skewed the error only surfaces at first run as
# `ModuleNotFoundError: Could not import module 'AutoModel'`.
RUN python - <<'PY'
import importlib, sys
mods = ["torch", "torchvision", "transformers", "tokenizers", "safetensors",
        "accelerate", "torchmetrics", "torchmetrics.image.fid",
        "transformers.AutoModel".rsplit(".", 1)[0]]
for m in mods:
    importlib.import_module(m)
from transformers import AutoModel  # the exact import that was failing
print("transformers stack OK:", AutoModel.__module__)
PY

# Copy the rest of the repo. Anything excluded by the launcher's staging
# (checkpoints/, wandb/, __pycache__/, .git/, .venv/) is already gone.
COPY . /workspace

# Pre-create the dirs train.py writes into so the first run doesn't race.
RUN mkdir -p /workspace/checkpoints

# Quick CUDA sanity check at build time (no-op if no GPU at build).
RUN python -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available())"

# Mixtrain invokes the MixFlow / MixModel class directly; no CMD needed.
# Kept for local debugging: `docker run -it <image> bash`.
CMD ["bash"]
