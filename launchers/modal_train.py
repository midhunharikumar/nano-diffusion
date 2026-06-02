"""
Launch nano-diffusion training on Modal.

Setup:
    pip install modal
    modal setup                                      # authenticate once
    modal secret create wandb-secret WANDB_API_KEY=<key>
    modal secret create huggingface-secret HF_TOKEN=<key>
    # optional — for GCS checkpoint uploads:
    modal secret create gcp-secret GOOGLE_APPLICATION_CREDENTIALS_JSON="$(cat key.json)"

Run:
    modal run launchers/modal_train.py
    modal run launchers/modal_train.py --dataset mnist
    modal run launchers/modal_train.py --dataset cifar10 --overrides "hidden_dim=512,depth=12,epochs=200"
    modal run launchers/modal_train.py --dataset imagenet64 --gcp_bucket checkpoints
    # to change GPU type, edit the gpu= arg in the @app.function decorator

Download checkpoints after run:
    modal volume get nano-diffusion-checkpoints . ./checkpoints
"""

from pathlib import Path

import modal

ROOT = Path(__file__).parent.parent

app = modal.App("nano-diffusion")

# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------
CUDA_TAG = "cu126"  # update if your Modal GPU needs a different build

image = (
    modal.Image.debian_slim(python_version="3.11")
    # gcc is required by Triton/Inductor (torch.compile) to build generated C code.
    # debian_slim strips build tools by default so we add them back explicitly.
    .apt_install("build-essential")
    # Install torch from the PyTorch wheel server only — prevents PyPI's generic
    # torch wheel (bundled with whatever CUDA it prefers) from winning on version.
    .pip_install(
        "torch",
        "torchvision",
        index_url=f"https://download.pytorch.org/whl/{CUDA_TAG}",
    )
    .pip_install(
        "datasets",
        "huggingface_hub",
        "einops",
        "omegaconf",
        "wandb",
        "Pillow",
        "tqdm",
        "torchmetrics[image]",
        "transformers",
        "accelerate",
        "google-cloud-storage",
    )
    # Cosmos tokenizer (only used when use_tokenizer=true); not on PyPI.
    .apt_install("git")
    .pip_install("cosmos-tokenizer @ git+https://github.com/NVIDIA/Cosmos-Tokenizer.git")
    # add_local_dir syncs at run time (no image rebuild needed on code changes)
    .add_local_dir(
        ROOT,
        remote_path="/app",
        ignore=[".git", "__pycache__", "checkpoints", "launchers", "*.pt",
                "exp_logs", "*.log"],  # live-written logs must not race the build sync
    )
)

# ---------------------------------------------------------------------------
# Volumes  (persist across runs)
# ---------------------------------------------------------------------------
checkpoints_vol = modal.Volume.from_name(
    "nano-diffusion-checkpoints", create_if_missing=True
)
hf_cache_vol = modal.Volume.from_name("nano-diffusion-hf-cache", create_if_missing=True)


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="H100",
    timeout=60 * 60 * 12,  # 12 h — adjust to your run budget
    volumes={
        "/checkpoints": checkpoints_vol,
        "/root/.cache/huggingface": hf_cache_vol,
    },
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("huggingface-secret"),
        # optional — only needed for GCS uploads; create with:
        #   modal secret create gcp-secret GOOGLE_APPLICATION_CREDENTIALS_JSON="$(cat key.json)"
        modal.Secret.from_name("gcp-secret"),
    ],
)
def train(
    dataset: str = "cifar10",
    run_name: str = "",
    overrides: list[str] = [],
    max_runtime: int = 0,
    gcp_bucket: str = "",
):
    import json
    import os
    import subprocess
    import sys

    os.chdir("/app")

    # redirect checkpoint dir to the persistent volume
    if not os.path.exists("checkpoints"):
        os.symlink("/checkpoints", "checkpoints")

    # cache Cosmos tokenizer .jit checkpoints on the persistent HF volume
    os.environ.setdefault("COSMOS_CACHE_DIR", "/root/.cache/huggingface/cosmos")

    # materialise GCP service-account key from the Modal secret env var
    creds_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON", "")
    if creds_json:
        creds_path = "/tmp/gcp_credentials.json"
        with open(creds_path, "w") as f:
            f.write(creds_json)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path

    cmd = [sys.executable, "train.py", f"configs/{dataset}.yaml", "device=cuda"]
    if run_name:
        cmd += ["--run_name", run_name]
    if max_runtime > 0:
        cmd += ["--max_runtime", str(max_runtime)]
    if gcp_bucket:
        cmd += ["--gcp_bucket", gcp_bucket]
    cmd += overrides
    subprocess.run(cmd, check=True)

    checkpoints_vol.commit()  # flush volume writes before container exits


# ---------------------------------------------------------------------------
# Tokenizer reconstruction check (no diffusion model)
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 30,
    volumes={
        "/checkpoints": checkpoints_vol,
        "/root/.cache/huggingface": hf_cache_vol,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def recon(dataset: str = "imagenet256_cosmos_di", n: int = 16):
    import os
    import subprocess
    import sys

    os.chdir("/app")
    if not os.path.exists("checkpoints"):
        os.symlink("/checkpoints", "checkpoints")
    os.environ.setdefault("COSMOS_CACHE_DIR", "/root/.cache/huggingface/cosmos")

    subprocess.run(
        [sys.executable, "recon.py", f"configs/{dataset}.yaml", "--n", str(n),
         "--out", f"checkpoints/recon_{dataset}.png"],
        check=True,
    )
    checkpoints_vol.commit()  # persist the comparison PNG


# ---------------------------------------------------------------------------
# Local entry point
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    dataset: str = "cifar10",
    run_name: str = "",  # base name; UUID suffix is appended automatically
    overrides: str = "",  # comma-separated dotlist, e.g. "hidden_dim=512,epochs=200"
    max_runtime: int = 0,  # stop after this many seconds (0 = no limit)
    gcp_bucket: str = "",  # GCS bucket name for checkpoint uploads
):
    train.remote(
        dataset=dataset,
        run_name=run_name,
        overrides=[o.strip() for o in overrides.split(",") if o.strip()],
        max_runtime=max_runtime,
        gcp_bucket=gcp_bucket,
    )

# Tokenizer reconstruction check (no extra entrypoint, so bare `modal run
# launchers/modal_train.py` still defaults to training):
#   modal run launchers/modal_train.py::recon --dataset imagenet256_cosmos_di --n 16
