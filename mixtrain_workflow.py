"""Mixtrain workflow wrapping nano-diffusion's train.py.

Deployed via:
    python mixtrain_launcher.py train --config cifar10 --epochs 5 --gpu A100

The whole repo (model.py, data.py, sample.py, eval.py, configs/, ...) ships as
the workflow source so train.py can run unchanged inside the sandbox.
"""
from __future__ import annotations

import glob
import os
import re
import subprocess
import sys
from pathlib import Path

from mixtrain import MixFlow, Sandbox
from mixtrain.types import JSON, Markdown

REPO = Path(__file__).resolve().parent


def _fid_from_name(p: str) -> float:
    m = re.search(r"fid([0-9.]+)", p or "")
    return float(m.group(1)) if m else float("inf")


class TrainDiffusion(MixFlow):
    """Train a flow-matching DiT on MNIST / CIFAR-10 / ImageNet inside Mixtrain.

    Thin wrapper around train.py: builds the argv (`config + --run_name +
    --max_runtime + --gcp_bucket + dotlist overrides`), shells out, then scans
    `checkpoints/` for the best FID and returns it as the workflow output.
    """

    _sandbox = Sandbox(
        image="pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime",
        gpu="A100",
        memory=40960,
        timeout=86400,   # hard cap; use max_runtime to stop earlier
    )

    def setup(self):
        """Install repo deps at boot. Pinned + extras prevent the
        `Could not import module 'AutoModel'` failure inside torchmetrics."""
        req = REPO / "requirements.txt"
        if req.exists():
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-q",
                "--upgrade", "--no-cache-dir",
                "-r", str(req),
            ])
        # Sanity-check the transformers stack up-front so failures surface
        # here (clear traceback) instead of inside torchmetrics later.
        from transformers import AutoModel  # noqa: F401

    def run(
        self,
        config: str = "cifar10",
        overrides: list[str] | None = None,
        run_name: str | None = None,
        max_runtime: int | None = None,
        gcp_bucket: str = "",
        wandb_api_key: str = "",
    ) -> dict:
        """Run nano-diffusion training.

        Args:
            config: Preset name (mnist, cifar10, cifar10_256, imagenet64,
                imagenet256) or a path to a YAML config.
            overrides: OmegaConf dotlist strings, e.g.
                ["hidden_dim=512", "epochs=20", "batch_size=128"].
            run_name: Override the config's run_name (UUID suffix still added).
            max_runtime: Stop training after N seconds and checkpoint cleanly.
            gcp_bucket: GCS bucket name for checkpoint upload (optional).
            wandb_api_key: If empty, W&B is disabled.

        Returns:
            best_fid, checkpoint path, all checkpoint paths, training exit code.
        """
        # resolve preset name → configs/<name>.yaml, else treat as a path
        cfg_path = REPO / "configs" / f"{config}.yaml"
        if not cfg_path.exists():
            cfg_path = Path(config)
        assert cfg_path.exists(), f"config not found: {config!r}"

        env = os.environ.copy()
        if wandb_api_key:
            env["WANDB_API_KEY"] = wandb_api_key
        else:
            env["WANDB_MODE"] = "disabled"

        cmd: list[str] = [sys.executable, str(REPO / "train.py"), str(cfg_path)]
        if run_name:
            cmd += ["--run_name", run_name]
        if max_runtime:
            cmd += ["--max_runtime", str(max_runtime)]
        if gcp_bucket:
            cmd += ["--gcp_bucket", gcp_bucket]
        # dotlist overrides pass through as positional args (parse_known_args)
        if overrides:
            cmd += list(overrides)

        print(f"$ {' '.join(cmd)}", flush=True)
        proc = subprocess.run(cmd, cwd=str(REPO), env=env, check=False)

        # find best checkpoint by parsing fid<NUMBER> from filenames
        ckpts = sorted(glob.glob(str(REPO / "checkpoints" / "*.pt")))
        best = min(ckpts, key=_fid_from_name, default=None)
        best_fid = _fid_from_name(best) if best else None

        return {
            "exit_code": proc.returncode,
            "checkpoint": str(best) if best else None,
            "best_fid": best_fid,
            "all_checkpoints": JSON(data=ckpts),
            "report": Markdown(content=(
                "# Diffusion training complete\n\n"
                f"- **Config:** `{cfg_path.name}`\n"
                f"- **Overrides:** `{overrides or []}`\n"
                f"- **Best FID:** {best_fid}\n"
                f"- **Checkpoint:** `{best}`\n"
            )),
        }
