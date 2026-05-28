#!/usr/bin/env python3
"""Mixtrain launcher for nano-diffusion.

Drives the flow-matching DiT training pipeline from one CLI:

    # train CIFAR-10 with overrides on an A100 for 2 hours
    python mixtrain_launcher.py train \\
        --config cifar10 \\
        --override hidden_dim=512 --override epochs=20 \\
        --gpu A100 --max-runtime 7200

    # deploy a checkpoint as an inference endpoint
    python mixtrain_launcher.py deploy-model \\
        --checkpoint checkpoints/cifar10_step0010000_fid12.50.pt \\
        --name nano-cifar10

    # generate samples from the deployed model
    python mixtrain_launcher.py sample --model nano-cifar10 --class-label 7

    # full pipeline: train → deploy best checkpoint → sample → eval
    python mixtrain_launcher.py launch \\
        --config mnist --override epochs=2 --gpu A100
"""
from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterator

REPO = Path(__file__).resolve().parent
CONFIG_PRESETS = ("mnist", "cifar10", "cifar10_256", "imagenet64", "imagenet256")

# directories/files to skip when bundling the repo as workflow/model source
_BUNDLE_EXCLUDES = shutil.ignore_patterns(
    "checkpoints", "wandb", "__pycache__", ".git", ".venv", "venv",
    "outputs", "*.pyc", ".DS_Store", ".pytest_cache", ".mypy_cache",
)


@contextlib.contextmanager
def _stage_repo(extra_files: dict[str, Path] | None = None) -> Iterator[Path]:
    """Copy the repo into a temp dir (minus heavy local junk).

    Args:
        extra_files: mapping of `dest_name → src_path` to copy into the staging
            dir alongside the source (used to bundle a checkpoint with the model).
    """
    with tempfile.TemporaryDirectory(prefix="mixtrain-stage-") as td:
        staging = Path(td) / REPO.name
        shutil.copytree(REPO, staging, ignore=_BUNDLE_EXCLUDES)
        for dest_name, src in (extra_files or {}).items():
            shutil.copy2(src, staging / dest_name)
        yield staging


def _run_cli(*argv: str) -> None:
    """Invoke the `mixtrain` CLI and propagate failures."""
    print("$ " + " ".join(argv))
    subprocess.run(list(argv), check=True)


# ---------- SDK lazy import (so `--help` works without mixtrain installed) ----------

def _sdk() -> dict[str, Any]:
    try:
        import mixtrain  # noqa: F401
    except ImportError:
        sys.stderr.write("error: mixtrain SDK not installed. Run: pip install mixtrain\n")
        sys.exit(2)
    from mixtrain import (
        Dataset, Eval, MixClient, Model, Workflow,
        generate_name, validate_resource_name,
    )
    return dict(
        Dataset=Dataset, Eval=Eval, MixClient=MixClient, Model=Model,
        Workflow=Workflow, generate_name=generate_name,
        validate_resource_name=validate_resource_name,
    )


def _print(payload: Any) -> None:
    print(json.dumps(payload, indent=2, default=str))


def _resolve_config(cfg: str) -> Path:
    """Accept either a preset name (`cifar10`) or a path."""
    p = REPO / "configs" / f"{cfg}.yaml"
    if p.exists():
        return p
    p2 = Path(cfg).expanduser().resolve()
    if p2.exists():
        return p2
    sys.exit(f"error: config '{cfg}' is not a preset ({CONFIG_PRESETS}) or a file path")


# ---------- commands ----------

def cmd_train(args: argparse.Namespace) -> None:
    """Deploy the TrainDiffusion workflow and run it."""
    sdk = _sdk()
    cfg_path = _resolve_config(args.config)
    workflow_name = args.name or f"nano-diffusion-{args.config}"
    sdk["validate_resource_name"](workflow_name, "workflow")

    # Deploy via the documented CLI: `mixtrain workflow create <dir> --name <name>`
    if not args.skip_deploy:
        with _stage_repo() as staging:
            print(f"→ deploying workflow '{workflow_name}' (source: {staging})")
            _run_cli("mixtrain", "workflow", "create", str(staging), "--name", workflow_name)

    inputs: dict[str, Any] = {
        "config": str(cfg_path.relative_to(REPO)) if cfg_path.is_relative_to(REPO) else str(cfg_path),
        "overrides": list(args.override or []),
    }
    if args.run_name:
        inputs["run_name"] = args.run_name
    if args.max_runtime:
        inputs["max_runtime"] = args.max_runtime
    if args.gcp_bucket:
        inputs["gcp_bucket"] = args.gcp_bucket
    if args.wandb_api_key:
        inputs["wandb_api_key"] = args.wandb_api_key

    sandbox: dict[str, Any] = {}
    if args.gpu:
        sandbox["gpu"] = args.gpu
    if args.timeout:
        sandbox["timeout"] = args.timeout

    wf = sdk["Workflow"](workflow_name)
    print(f"→ running workflow with inputs={inputs} sandbox={sandbox or None}")
    if args.detach:
        info = wf.submit(inputs=inputs, sandbox=sandbox or None)
        _print(info)
        return

    result = wf.run(inputs=inputs, sandbox=sandbox or None)
    outputs = result.get("outputs", {}) if isinstance(result, dict) else {}
    _print({
        "status": result.get("status") if isinstance(result, dict) else str(result),
        "best_fid": outputs.get("best_fid"),
        "checkpoint": outputs.get("checkpoint"),
    })


def cmd_deploy_model(args: argparse.Namespace) -> None:
    """Deploy DiffusionSampler with a checkpoint bundled as model_weights.pt."""
    sdk = _sdk()
    sdk["validate_resource_name"](args.name, "model")

    ckpt = Path(args.checkpoint).expanduser().resolve()
    if not ckpt.exists():
        sys.exit(f"error: checkpoint not found: {ckpt}")

    # Stage the repo + drop the checkpoint into it as model_weights.pt.
    # DiffusionSampler.setup() loads from that path by default.
    with _stage_repo(extra_files={"model_weights.pt": ckpt}) as staging:
        print(f"→ deploying model '{args.name}' (source: {staging}, checkpoint: {ckpt.name})")
        _run_cli("mixtrain", "model", "create", str(staging), "--name", args.name)
    print(f"✓ model '{args.name}' deployed")


def cmd_sample(args: argparse.Namespace) -> None:
    """Generate samples by invoking the deployed sampler model."""
    sdk = _sdk()
    inputs: dict[str, Any] = {
        "class_label": args.class_label,
        "n_images": args.n_images,
        "n_steps": args.n_steps,
        "cfg_scale": args.cfg_scale,
    }
    model = sdk["Model"](args.model)
    print(f"→ sampling from '{args.model}' inputs={inputs}")
    result = model.run(inputs=inputs)
    out = result.outputs if hasattr(result, "outputs") else result
    _print({
        "class_label": out.get("class_label"),
        "n_images": len(out.get("images", [])),
        "first_image_url": (out.get("images") or [{}])[0].get("url", "")[:80] + "...",
    })


def cmd_eval(args: argparse.Namespace) -> None:
    """Build a comparison Eval over a sample-dataset of generations."""
    sdk = _sdk()
    eval_name = args.name or sdk["generate_name"](args.dataset, "eval")
    sdk["validate_resource_name"](eval_name, "eval")

    ds = sdk["Dataset"](args.dataset)
    if sdk["Eval"].exists(eval_name):
        print(f"  eval '{eval_name}' already exists — reusing")
        ev = sdk["Eval"](eval_name)
    else:
        cols = args.columns.split(",") if args.columns else None
        ev = sdk["Eval"].from_dataset(ds, name=eval_name, columns=cols)
    _print({"eval": eval_name, "dataset": args.dataset, "url": getattr(ev, "url", None)})


def cmd_sweep_eval(args: argparse.Namespace) -> None:
    """Sample one image per class, save as a dataset, and build an eval over it."""
    sdk = _sdk()
    n_classes = args.num_classes
    model = sdk["Model"](args.model)

    rows = []
    for c in range(n_classes):
        result = model.run(inputs={
            "class_label": c, "n_images": 1,
            "n_steps": args.n_steps, "cfg_scale": args.cfg_scale,
        })
        out = result.outputs if hasattr(result, "outputs") else result
        rows.append({"class_label": c, "sample": out["images"][0]})

    import pandas as pd  # heavy but ok for CLI side
    from mixtrain import Image as MixImage  # type: ignore
    df = pd.DataFrame(rows)
    ds_name = args.dataset_name
    ds = sdk["Dataset"].from_pandas(df).save(
        ds_name,
        description=f"Class sweep from {args.model}",
        column_types={"sample": MixImage},
    )
    print(f"  saved dataset '{ds_name}' ({len(rows)} rows)")
    eval_name = args.eval_name or sdk["generate_name"](args.model, "eval")
    sdk["Eval"].from_dataset(ds, name=eval_name)
    _print({"eval": eval_name, "dataset": ds_name})


def cmd_launch(args: argparse.Namespace) -> None:
    """End-to-end: train → deploy-model → class-sweep eval."""
    project = args.project or args.config
    workflow_name = f"{project}-train"
    model_name = f"{project}-sampler"

    # 1. train
    train_args = argparse.Namespace(
        config=args.config,
        override=args.override,
        name=workflow_name,
        run_name=None,
        max_runtime=args.max_runtime,
        gcp_bucket="",
        wandb_api_key=args.wandb_api_key,
        gpu=args.gpu,
        timeout=args.timeout,
        detach=False,
        skip_deploy=False,
    )
    cmd_train(train_args)

    # 2. find best checkpoint locally (the workflow surfaces its path)
    sdk = _sdk()
    runs = sdk["Workflow"](workflow_name).runs
    last = runs[-1] if runs else None
    ckpt = (last and last.get("outputs", {}).get("checkpoint")) if isinstance(last, dict) else None
    if not ckpt:
        sys.exit("error: training did not produce a checkpoint; cannot deploy model")

    # 3. deploy model
    cmd_deploy_model(argparse.Namespace(name=model_name, checkpoint=ckpt))

    # 4. class-sweep eval
    cmd_sweep_eval(argparse.Namespace(
        model=model_name,
        dataset_name=f"{project}-samples",
        eval_name=f"{project}-eval",
        num_classes=args.num_classes,
        n_steps=args.n_steps,
        cfg_scale=args.cfg_scale,
    ))

    print(f"=== done. workflow={workflow_name} model={model_name} ===")


# ---------- arg parser ----------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mixtrain-launcher",
        description="Launcher for nano-diffusion on Mixtrain.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # train -------------------------------------------------------------
    tr = sub.add_parser("train", help="Train a DiT on Mixtrain (wraps train.py)")
    tr.add_argument(
        "--config", required=True,
        help=f"Preset name {CONFIG_PRESETS} or path to YAML",
    )
    tr.add_argument(
        "--override", action="append", default=[], metavar="KEY=VALUE",
        help="OmegaConf dotlist override; repeatable",
    )
    tr.add_argument("--name", default=None, help="Workflow name")
    tr.add_argument("--run-name", dest="run_name", default=None,
                    help="Run name passed to train.py")
    tr.add_argument("--max-runtime", dest="max_runtime", type=int, default=None,
                    metavar="SECONDS", help="train.py --max_runtime")
    tr.add_argument("--gcp-bucket", dest="gcp_bucket", default="",
                    help="GCS bucket for checkpoint upload")
    tr.add_argument("--wandb-api-key", dest="wandb_api_key", default="",
                    help="W&B API key (empty disables W&B)")
    tr.add_argument("--gpu", default="A100", help="Sandbox GPU type")
    tr.add_argument("--timeout", type=int, default=None,
                    help="Sandbox timeout seconds")
    tr.add_argument("--detach", action="store_true", help="Submit and exit")
    tr.add_argument("--skip-deploy", dest="skip_deploy", action="store_true",
                    help="Skip `mixtrain workflow create`; run existing workflow")
    tr.set_defaults(func=cmd_train)

    # deploy-model ------------------------------------------------------
    dm = sub.add_parser("deploy-model", help="Deploy a checkpoint as a MixModel")
    dm.add_argument("--checkpoint", required=True, help="Path to a .pt file")
    dm.add_argument("--name", required=True, help="Model name")
    dm.set_defaults(func=cmd_deploy_model)

    # sample ------------------------------------------------------------
    sm = sub.add_parser("sample", help="Generate samples from a deployed model")
    sm.add_argument("--model", required=True, help="Deployed model name")
    sm.add_argument("--class-label", dest="class_label", type=int, default=0)
    sm.add_argument("--n-images", dest="n_images", type=int, default=1)
    sm.add_argument("--n-steps", dest="n_steps", type=int, default=50)
    sm.add_argument("--cfg-scale", dest="cfg_scale", type=float, default=3.0)
    sm.set_defaults(func=cmd_sample)

    # eval (over existing dataset) --------------------------------------
    ev = sub.add_parser("eval", help="Create an Eval view from a dataset")
    ev.add_argument("--dataset", required=True)
    ev.add_argument("--name", default=None)
    ev.add_argument("--columns", default=None, help="Comma-separated columns")
    ev.set_defaults(func=cmd_eval)

    # sweep-eval --------------------------------------------------------
    sw = sub.add_parser(
        "sweep-eval",
        help="Sample one image per class, save as dataset + eval",
    )
    sw.add_argument("--model", required=True)
    sw.add_argument("--num-classes", dest="num_classes", type=int, default=10)
    sw.add_argument("--dataset-name", dest="dataset_name", required=True)
    sw.add_argument("--eval-name", dest="eval_name", default=None)
    sw.add_argument("--n-steps", dest="n_steps", type=int, default=50)
    sw.add_argument("--cfg-scale", dest="cfg_scale", type=float, default=3.0)
    sw.set_defaults(func=cmd_sweep_eval)

    # launch ------------------------------------------------------------
    la = sub.add_parser("launch", help="Train → deploy → sweep-eval pipeline")
    la.add_argument("--config", required=True)
    la.add_argument("--override", action="append", default=[])
    la.add_argument("--project", default=None, help="Naming prefix (default: config)")
    la.add_argument("--max-runtime", dest="max_runtime", type=int, default=None)
    la.add_argument("--gpu", default="A100")
    la.add_argument("--timeout", type=int, default=None)
    la.add_argument("--wandb-api-key", dest="wandb_api_key", default="")
    la.add_argument("--num-classes", dest="num_classes", type=int, default=10)
    la.add_argument("--n-steps", dest="n_steps", type=int, default=50)
    la.add_argument("--cfg-scale", dest="cfg_scale", type=float, default=3.0)
    la.set_defaults(func=cmd_launch)

    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
