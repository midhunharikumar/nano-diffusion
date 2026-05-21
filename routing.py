"""
Semantic Routing — arxiv 2602.03510

Instead of conditioning the DiT on only the last LLM layer, extract ALL layer
hidden states and learn per-DiT-depth weights to fuse them (depth-wise strategy S2).

Architecture:
  text prompt
       │
  LLMTextEncoder          ← frozen; outputs hidden states from every layer
       │
  [h_0, h_1, …, h_L]     ← one tensor (B, S, llm_d) per LLM layer
       │
  DepthwiseRouter         ← for each DiT block depth d:
       │                       weights_d = softmax(learnable_d)  shape (L,)
       │                       fused_d   = Σ weights_d[l] * LN(h_l)
       │                       fused_d   = proj(fused_d)          (B, S, dit_d)
       ↓
  cross-attention in each DiT block

Two supported LLM backends:
  google/gemma-3-1b           (gemma3:270m Ollama ≈ google/gemma-3-270m if released)
  Qwen/Qwen3-0.6B             (qwen3.5:0.8b Ollama ≈ smallest Qwen3 on HF)

Usage:
  encoder = LLMTextEncoder("google/gemma-3-1b")
  router  = DepthwiseRouter(encoder.num_layers, num_dit_layers=6,
                             llm_hidden_size=encoder.hidden_size, dit_hidden_size=256)
  hidden_states = encoder.encode(["a red truck", "a cat"], device)
  fused = router.fuse(hidden_states, dit_depth=3)   # (B, S, 256)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Text descriptions for each class label — richer than bare integers
CLASS_TEXTS = {
    "mnist": [
        "handwritten digit zero",  "handwritten digit one",  "handwritten digit two",
        "handwritten digit three", "handwritten digit four", "handwritten digit five",
        "handwritten digit six",   "handwritten digit seven","handwritten digit eight",
        "handwritten digit nine",
    ],
    "cifar10": [
        "an airplane", "an automobile", "a bird", "a cat", "a deer",
        "a dog", "a frog", "a horse", "a ship", "a truck",
    ],
}

# HuggingFace model IDs for each shorthand
LLM_MODEL_IDS = {
    "gemma3":  "google/gemma-3-1b",       # smallest public Gemma 3 on HF
    "qwen3":   "Qwen/Qwen3-0.6B",         # smallest Qwen3 on HF
}


# ---------------------------------------------------------------------------
# Text encoder (frozen LLM — all layers)
# ---------------------------------------------------------------------------

class LLMTextEncoder(nn.Module):
    """Frozen LLM that returns hidden states from every transformer layer."""

    def __init__(self, model_name: str):
        super().__init__()
        from transformers import AutoTokenizer, AutoModel
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.hidden_size = self.model.config.hidden_size
        # +1: embedding layer output is included in hidden_states tuple
        self.num_layers = self.model.config.num_hidden_layers + 1

    @torch.no_grad()
    def encode(self, texts: list[str], device) -> list[torch.Tensor]:
        """
        Returns list of num_layers tensors, each (B, S, hidden_size).
        Index 0 = embedding output, index L = last transformer layer.
        """
        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True,
            truncation=True, max_length=64,
        ).to(device)
        outputs = self.model(**inputs, output_hidden_states=True)
        return list(outputs.hidden_states)


# ---------------------------------------------------------------------------
# Depth-wise router (strategy S2 from paper — best trade-off)
# ---------------------------------------------------------------------------

class DepthwiseRouter(nn.Module):
    """
    Per-DiT-block learnable weights over LLM layers.
    Zero-init → starts as uniform average; learns to specialise per depth.
    """

    def __init__(
        self,
        num_llm_layers:  int,
        num_dit_layers:  int,
        llm_hidden_size: int,
        dit_hidden_size: int,
    ):
        super().__init__()
        self.llm_hidden_size = llm_hidden_size

        # one weight vector (L,) per DiT block — zero-init = uniform softmax start
        self.layer_weights = nn.ParameterList([
            nn.Parameter(torch.zeros(num_llm_layers))
            for _ in range(num_dit_layers)
        ])

        # project LLM dim → DiT dim; zero-init output so cross-attn starts silent
        self.proj = nn.Linear(llm_hidden_size, dit_hidden_size)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def fuse(self, text_hidden_states: list[torch.Tensor], dit_depth: int) -> torch.Tensor:
        """
        text_hidden_states : list of L tensors, each (B, S, llm_hidden_size)
        dit_depth          : which DiT block is calling (selects weight vector)
        returns            : (B, S, dit_hidden_size)
        """
        stacked = torch.stack(text_hidden_states, dim=0).float()       # (L, B, S, C)
        stacked = F.layer_norm(stacked, (self.llm_hidden_size,))        # per-layer LN
        weights = F.softmax(self.layer_weights[dit_depth], dim=0)       # (L,)
        fused   = torch.einsum("l,lbsc->bsc", weights, stacked)         # (B, S, C_llm)
        return self.proj(fused)                                          # (B, S, C_dit)
