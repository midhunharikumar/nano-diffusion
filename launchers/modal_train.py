"""
Launch nano-diffusion training on Modal.

Setup:
    pip install modal
    modal setup                                      # authenticate once
    modal secret create wandb-secret WANDB_API_KEY=<key>

Run:
    modal run launchers/modal_train.py
    modal run launchers/modal_train.py --dataset mnist
    modal run launchers/modal_train.py --dataset cifar10 --overrides "hidden_dim=512,depth=12,epochs=200"
    # to change GPU type, edit the gpu= arg in the @app.function decorator

Download checkpoints after run:
    modal volume get nano-diffusion-checkpoints . ./checkpoints
"""

import modal
from pathlib import Path

ROOT = Path(__file__).parent.parent

app = modal.App("nano-diffusion")

# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "torchvision",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "datasets", "huggingface_hub", "einops",
        "omegaconf", "wandb", "Pillow", "tqdm",
        "torchmetrics[image]",
    )
    # add_local_dir syncs at run time (no image rebuild needed on code changes)
    .add_local_dir(
        ROOT,
        remote_path="/app",
        ignore=[".git", "__pycache__", "checkpoints", "launchers", "*.pt"],
    )
)

# ---------------------------------------------------------------------------
# Volumes  (persist across runs)
# ---------------------------------------------------------------------------
checkpoints_vol = modal.Volume.from_name("nano-diffusion-checkpoints", create_if_missing=True)
hf_cache_vol    = modal.Volume.from_name("nano-diffusion-hf-cache",    create_if_missing=True)


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 60 * 12,   # 12 h — adjust to your run budget
    volumes={
        "/checkpoints":             checkpoints_vol,
        "/root/.cache/huggingface": hf_cache_vol,
    },
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def train(dataset: str = "cifar10", overrides: list[str] = []):
    import os, subprocess, sys

    os.chdir("/app")

    # redirect checkpoint dir to the persistent volume
    if not os.path.exists("checkpoints"):
        os.symlink("/checkpoints", "checkpoints")

    cmd = [sys.executable, "train.py", f"configs/{dataset}.yaml", "device=cuda"] + overrides
    subprocess.run(cmd, check=True)

    checkpoints_vol.commit()   # flush volume writes before container exits


# ---------------------------------------------------------------------------
# Local entry point
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    dataset:  str = "cifar10",
    overrides: str = "",   # comma-separated dotlist, e.g. "hidden_dim=512,epochs=200"
):
    train.remote(
        dataset=dataset,
        overrides=[o.strip() for o in overrides.split(",") if o.strip()],
    )
