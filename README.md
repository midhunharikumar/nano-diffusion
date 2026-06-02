# nano-diffusion

Minimal flow-matching diffusion model on MNIST / CIFAR-10. DiT backbone, classifier-free guidance, optional semantic routing.

## Install

```bash
pip install -r requirements.txt
```

## Train locally

```bash
python train.py configs/mnist.yaml
python train.py configs/cifar10.yaml

# overrides via dotlist
python train.py configs/cifar10.yaml hidden_dim=512 depth=12 epochs=200

# custom run name (UUID suffix is always appended)
python train.py configs/cifar10.yaml --run_name my-run
```

## Sample

```bash
python sample.py --checkpoint checkpoints/<run>_step0000500.pt --n_steps 50 --cfg_scale 3.0
```

## Train on Modal (H100)

```bash
pip install modal
modal setup                                        # authenticate once
modal secret create wandb-secret WANDB_API_KEY=<key>
modal secret create huggingface-secret HF_TOKEN=<token>

modal run launchers/modal_train.py
modal run launchers/modal_train.py --dataset mnist
modal run launchers/modal_train.py --dataset cifar10 --run-name exp1 --overrides "hidden_dim=512,depth=12,epochs=200"

# 256px (gradient checkpointing enabled by default in the config)
modal run launchers/modal_train.py --dataset cifar10_256

# download checkpoints after run
modal volume get nano-diffusion-checkpoints . ./checkpoints
```

## Key config options

| Key | Default | Description |
|---|---|---|
| `hidden_dim` / `depth` / `num_heads` | 256 / 6 / 4 | Model size |
| `batch_size` | 256 | Training batch size |
| `lr` | 1e-4 | Peak learning rate (cosine + 5% warmup) |
| `cfg_dropout` | 0.1 | Classifier-free guidance dropout rate |
| `cfg_scale` | 3.0 | CFG scale at sampling |
| `n_sample_steps` | 50 | Euler ODE steps |
| `eval_interval` | 500 | Steps between sample logging |
| `fid_every_n_epochs` | 5 | FID evaluation frequency |
| `checkpoint_every` | 0 | Gradient checkpointing (0=off, 1=all blocks, 2=every other) |
| `use_semantic_routing` | false | Enable LLM semantic routing (arxiv 2602.03510) |
| `llm_model_name` | — | HuggingFace model ID for routing (e.g. `Qwen/Qwen3-0.6B`) |
| `use_tokenizer` | false | Train in Cosmos latent space instead of pixels |
| `tokenizer_name` | `Cosmos-0.1-Tokenizer-CI8x8` | Cosmos continuous image tokenizer (`CI8x8` 8×, `CI16x16` 16×) |
| `latent_scale` / `latent_shift` | 1.0 / 0.0 | Latent normalisation `(z - shift) * scale` |

## Cosmos tokenizer (latent diffusion)

Train the DiT in the compressed latent space of NVIDIA's
[Cosmos tokenizer](https://github.com/NVIDIA/Cosmos-Tokenizer) instead of raw
pixels. The `CI8x8` continuous image tokenizer maps a 256px RGB image to a
32×32×4 latent grid (8× spatial compression), so the DiT runs on short
sequences while still producing high-resolution images.

```bash
# the tokenizer package is not on PyPI — install from GitHub
pip install git+https://github.com/NVIDIA/Cosmos-Tokenizer.git

# .jit checkpoints auto-download from HuggingFace on first use
python train.py configs/imagenet256_cosmos.yaml
```

Flow matching mixes latents with N(0, I) noise, so the latents should be roughly
unit-variance. Estimate per-dataset normalisation and set `latent_scale` /
`latent_shift` in the config:

```bash
python tokenizer.py configs/imagenet256_cosmos.yaml   # prints suggested scale/shift
```

Sampling and FID decode latents back to pixels automatically. `use_tokenizer`
and `use_reg` are mutually exclusive (REG needs pixel images for DINOv2).

## Semantic routing

Enable depth-wise LLM layer fusion ([arxiv 2602.03510](https://arxiv.org/abs/2602.03510)):

```bash
python train.py configs/cifar10.yaml use_semantic_routing=true llm_model_name=Qwen/Qwen3-0.6B
```

Sample with text prompts:

```bash
python sample.py --checkpoint <ckpt> --text "a red airplane"
```
