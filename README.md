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

## Semantic routing

Enable depth-wise LLM layer fusion ([arxiv 2602.03510](https://arxiv.org/abs/2602.03510)):

```bash
python train.py configs/cifar10.yaml use_semantic_routing=true llm_model_name=Qwen/Qwen3-0.6B
```

Sample with text prompts:

```bash
python sample.py --checkpoint <ckpt> --text "a red airplane"
```
