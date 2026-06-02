import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.checkpoint import checkpoint


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.g   = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.g


class SinusoidalEmbed(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.d = d

    def forward(self, t):
        t    = t * 1000.0
        half = self.d // 2
        freq = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
        args = t[:, None] * freq[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class SwiGLU(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        hidden = (int(d * 8 / 3) + 63) // 64 * 64
        self.gate = nn.Linear(d, hidden, bias=False)
        self.val  = nn.Linear(d, hidden, bias=False)
        self.proj = nn.Linear(hidden, d,  bias=False)

    def forward(self, x):
        return self.proj(F.silu(self.gate(x)) * self.val(x))


class ExpertChoiceRouter(nn.Module):
    """
    Expert Choice routing: each expert independently selects its top-k tokens.

    Guarantees perfect load balance by construction — no auxiliary loss needed.
    Gate scores are normalised with softmax over the selected token set per expert,
    so each expert's output is a weighted sum of its chosen tokens' FFN results.
    """

    def __init__(self, d: int, num_experts: int):
        super().__init__()
        self.gate = nn.Linear(d, num_experts, bias=False)

    def forward(self, x: torch.Tensor, capacity: int):
        """
        x        : (n_tokens, d)
        capacity : number of tokens each expert processes
        Returns (top_ids, top_weights) both (num_experts, capacity).
        """
        logits   = self.gate(x)                              # (n_tokens, E)
        scores   = logits.T                                  # (E, n_tokens)
        top_ids  = scores.topk(capacity, dim=1).indices      # (E, capacity)
        top_w    = scores.gather(1, top_ids).softmax(dim=1)  # (E, capacity)
        return top_ids, top_w


class MoEFFN(nn.Module):
    """
    Drop-in replacement for SwiGLU using Mixture-of-Experts with Expert Choice routing.

    num_always_on experts run densely on every token (guaranteed coverage).
    The remaining (num_experts - num_always_on) experts use Expert Choice: each
    selects its top-k tokens where k = floor(n_tokens * capacity_factor / n_routed).
    All expert outputs are summed into the residual stream.

    Parameters
    ----------
    num_experts     : total number of SwiGLU experts
    num_always_on   : experts that process every token (0 = fully sparse)
    capacity_factor : routed expert token budget relative to uniform share
                      (1.0 = exact fair share, 1.25 = 25% over-capacity)
    """

    def __init__(self, d: int, num_experts: int, num_always_on: int,
                 capacity_factor: float = 1.25):
        super().__init__()
        assert 0 <= num_always_on <= num_experts
        self.num_always_on   = num_always_on
        self.num_routed      = num_experts - num_always_on
        self.capacity_factor = capacity_factor
        self.experts         = nn.ModuleList([SwiGLU(d) for _ in range(num_experts)])
        self.router          = (ExpertChoiceRouter(d, self.num_routed)
                                if self.num_routed > 0 else None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, d = x.shape

        # Dense path: always-on experts see every token
        out = (sum(e(x) for e in self.experts[:self.num_always_on])
               if self.num_always_on > 0 else torch.zeros_like(x))

        if self.router is None:
            return out

        # Sparse path: each routed expert picks its top-capacity tokens
        x_flat   = x.reshape(B * T, d)
        capacity = max(1, int(B * T * self.capacity_factor / self.num_routed))
        top_ids, top_w = self.router(x_flat, capacity)   # (E, cap) each

        routed = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts[self.num_always_on:]):
            sel = x_flat[top_ids[i]]                                 # (cap, d)
            y   = expert(sel) * top_w[i].unsqueeze(-1)               # (cap, d)
            routed.scatter_add_(0, top_ids[i].unsqueeze(-1).expand_as(y), y)

        return out + routed.reshape(B, T, d)


class Attention(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.heads  = heads
        self.head_d = d // heads
        self.qkv    = nn.Linear(d, 3 * d, bias=False)
        self.proj   = nn.Linear(d, d,     bias=False)
        self.q_norm = RMSNorm(self.head_d)
        self.k_norm = RMSNorm(self.head_d)

    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [rearrange(t, "b n (h d) -> b h n d", h=self.heads) for t in (q, k, v)]
        q = self.q_norm(q)
        k = self.k_norm(k)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.proj(rearrange(out, "b h n d -> b n (h d)"))


class CrossAttention(nn.Module):
    """Cross-attention for attending image patches to routed text features."""

    def __init__(self, d: int, heads: int):
        super().__init__()
        self.heads  = heads
        self.head_d = d // heads
        self.q      = nn.Linear(d, d, bias=False)
        self.kv     = nn.Linear(d, 2 * d, bias=False)
        self.proj   = nn.Linear(d, d, bias=False)
        self.q_norm = RMSNorm(self.head_d)
        self.k_norm = RMSNorm(self.head_d)

    def forward(self, x, context):
        q      = rearrange(self.q(x), "b n (h d) -> b h n d", h=self.heads)
        k, v   = self.kv(context).chunk(2, dim=-1)
        k      = rearrange(k, "b n (h d) -> b h n d", h=self.heads)
        v      = rearrange(v, "b n (h d) -> b h n d", h=self.heads)
        q, k   = self.q_norm(q), self.k_norm(k)
        out    = F.scaled_dot_product_attention(q, k, v)
        return self.proj(rearrange(out, "b h n d -> b n (h d)"))


class TREADRouter:
    """
    TREAD token routing (compvis/tread).

    During training a random subset of image-patch tokens is "teleported"
    from route_start to route_end, bypassing those blocks entirely.
    At route_end the teleported tokens re-enter carrying their pre-route
    values; the processed tokens overwrite only their own positions.

    Training-only: when model.training is False the router is never called,
    so inference is identical to a plain DiT.

    The REG semantic token (position 0 when use_reg=True) is always excluded
    from routing and sees every block.
    """

    @staticmethod
    def get_ids(n_tokens: int, selection_rate: float, batch_size: int, device) -> torch.Tensor:
        """Random indices of tokens to KEEP (not teleport). Shape (B, n_keep)."""
        n_keep = n_tokens - int(n_tokens * selection_rate)
        return torch.argsort(torch.rand(batch_size, n_tokens, device=device), dim=1)[:, :n_keep]

    @staticmethod
    def gather(x: torch.Tensor, ids_keep: torch.Tensor) -> torch.Tensor:
        """Shrink sequence to kept tokens: (B, N, d) → (B, n_keep, d)."""
        return x.gather(1, ids_keep.unsqueeze(-1).expand(-1, -1, x.size(2)))

    @staticmethod
    def scatter(x_routed: torch.Tensor, ids_keep: torch.Tensor,
                x_saved: torch.Tensor) -> torch.Tensor:
        """Merge kept tokens back; teleported positions keep their pre-route values."""
        return x_saved.scatter(1, ids_keep.unsqueeze(-1).expand(-1, -1, x_saved.size(2)), x_routed)


class DiTBlock(nn.Module):
    def __init__(self, d: int, heads: int, cond_d: int, use_cross_attn: bool = False,
                 ffn: nn.Module = None):
        super().__init__()
        self.norm1 = RMSNorm(d)
        self.norm2 = RMSNorm(d)
        self.attn  = Attention(d, heads)
        self.ffn   = ffn if ffn is not None else SwiGLU(d)
        # AdaLN-Zero: (shift, scale, gate) × 2 for self-attn and FFN
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(cond_d, 6 * d))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

        # Optional cross-attention for semantic routing
        # gate init to 0 → tanh(0) = 0, so cross-attn starts silent
        if use_cross_attn:
            self.norm_ctx    = RMSNorm(d)
            self.cross_attn  = CrossAttention(d, heads)
            self.cross_gate  = nn.Parameter(torch.zeros(1))
        else:
            self.cross_attn = None

    def forward(self, x, cond, context=None):
        s1, a1, g1, s2, a2, g2 = self.ada(cond).chunk(6, dim=-1)
        x = x + g1[:, None] * self.attn(self.norm1(x) * (1 + a1[:, None]) + s1[:, None])
        if self.cross_attn is not None and context is not None:
            x = x + torch.tanh(self.cross_gate) * self.cross_attn(self.norm_ctx(x), context)
        x = x + g2[:, None] * self.ffn( self.norm2(x) * (1 + a2[:, None]) + s2[:, None])
        return x


class DiT(nn.Module):
    def __init__(
        self,
        img_size:    int  = 32,
        patch_size:  int  = 4,
        channels:    int  = 1,
        num_classes: int  = 10,
        d:           int  = 256,
        depth:       int  = 6,
        heads:       int  = 4,
        # semantic routing
        use_semantic_routing: bool = False,
        llm_model_name:       str  = None,
        checkpoint_every:     int  = 0,   # 0=off, 1=all blocks, N=every Nth block
        # REG: representation entanglement (arxiv 2507.01467)
        use_reg:       bool = False,
        reg_model_name: str = "facebook/dinov2-base",
        # TREAD: token routing for training efficiency (compvis/tread)
        use_tread:            bool  = False,
        tread_selection_rate: float = 0.5,   # fraction of patch tokens to teleport
        tread_route_start:    int   = 2,     # block index where route begins
        tread_route_end:      int   = -1,    # block index where route ends; -1 → depth-4
        # MaskGIT-style token dropout (training-only; REG semantic token excluded)
        use_maskgit:          bool  = False,
        maskgit_ratio:        float = 0.5,   # fraction of patch tokens to drop each step
        # MoE FFN with Expert Choice routing
        use_moe:              bool  = False,
        moe_num_experts:      int   = 8,     # total experts per MoE block
        moe_num_always_on:    int   = 1,     # always-on (dense) experts; rest are routed
        moe_capacity_factor:  float = 1.25,  # each routed expert's token budget vs fair share
        moe_every_n:          int   = 1,     # replace FFN with MoE every Nth block (1 = all)
        # Codebook cross-entropy: auxiliary classification head predicting the
        # tokenizer's discrete FSQ levels from the patch features (arxiv 2501.03575
        # tokenizer; mixed continuous-diffusion + discrete-code objective).
        use_codebook_ce:      bool  = False,
        fsq_levels:    list[int] | None = None,  # per-dim FSQ level counts
        ce_output:            bool  = False,  # CE-only ablation: x0 from argmax(code logits)
    ):
        super().__init__()
        assert d % heads == 0, (
            f"hidden_dim ({d}) must be divisible by num_heads ({heads}); "
            f"head_dim would be {d}/{heads} = {d/heads:.2f}"
        )
        self.img_size   = img_size
        self.patch_size = patch_size
        self.channels   = channels
        self.depth      = depth
        self.use_semantic_routing = use_semantic_routing
        self.checkpoint_every     = checkpoint_every
        n_patches = (img_size // patch_size) ** 2
        patch_dim = channels * patch_size * patch_size

        self.patch_embed = nn.Linear(patch_dim, d)
        self.pos_embed   = nn.Parameter(torch.zeros(1, n_patches, d))
        nn.init.normal_(self.pos_embed, std=0.02)

        self.t_embed = nn.Sequential(
            SinusoidalEmbed(d),
            nn.Linear(d, 4 * d), nn.SiLU(), nn.Linear(4 * d, d),
        )
        # +1 = null token for CFG (used in both class-conditional and routing modes)
        self.cls_embed = nn.Embedding(num_classes + 1, d)

        self.blocks = nn.ModuleList([
            DiTBlock(
                d, heads, d, use_cross_attn=use_semantic_routing,
                ffn=(MoEFFN(d, moe_num_experts, moe_num_always_on, moe_capacity_factor)
                     if use_moe and i % moe_every_n == 0 else None),
            )
            for i in range(depth)
        ])
        self.norm_out = RMSNorm(d)
        self.out      = nn.Linear(d, patch_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

        # Semantic routing modules (optional)
        if use_semantic_routing:
            from routing import LLMTextEncoder, DepthwiseRouter
            self.text_encoder = LLMTextEncoder(llm_model_name)
            self.router = DepthwiseRouter(
                num_llm_layers  = self.text_encoder.num_layers,
                num_dit_layers  = depth,
                llm_hidden_size = self.text_encoder.hidden_size,
                dit_hidden_size = d,
            )
        else:
            self.text_encoder = None
            self.router       = None

        # REG: entangle a DINOv2 CLS token with the patch sequence (arxiv 2507.01467)
        self.use_reg = use_reg
        if use_reg:
            from reg import REGEncoder
            self.reg_encoder = REGEncoder(reg_model_name)
            reg_d = self.reg_encoder.hidden_size
            # project DINOv2 space → DiT space; zero-init → starts silent
            self.sem_proj = nn.Linear(reg_d, d)
            nn.init.zeros_(self.sem_proj.weight)
            nn.init.zeros_(self.sem_proj.bias)
            # separate head to predict the clean DINOv2 CLS token; zero-init
            self.sem_out  = nn.Linear(d, reg_d)
            nn.init.zeros_(self.sem_out.weight)
            nn.init.zeros_(self.sem_out.bias)
        else:
            self.reg_encoder = None
            self.sem_proj    = None
            self.sem_out     = None

        # TREAD: token routing (training-only, no extra parameters)
        self.use_tread            = use_tread
        self.tread_selection_rate = tread_selection_rate
        self.tread_route_start    = tread_route_start
        # depth - 4 is the paper's recommended end point; clamp so start < end
        self.tread_route_end      = (depth - 4 if tread_route_end < 0
                                     else tread_route_end)

        # MaskGIT: drop patch tokens for the entire forward pass (training-only)
        self.use_maskgit   = use_maskgit
        self.maskgit_ratio = maskgit_ratio

        # Codebook CE: parallel head mapping each patch token to per-FSQ-dim level
        # logits for every latent position it covers. Zero-init → starts silent.
        self.use_codebook_ce = use_codebook_ce
        self.fsq_levels      = list(fsq_levels) if fsq_levels else None
        # CE-only ablation: drive sampling from the classification head (argmax →
        # per-dim level → code value) instead of the regression head.
        self.ce_output       = ce_output
        if use_codebook_ce:
            assert not use_maskgit, "codebook CE not supported with MaskGIT (patch-space output)"
            assert self.fsq_levels, "use_codebook_ce requires fsq_levels"
            self.code_head = nn.Linear(d, patch_size * patch_size * sum(self.fsq_levels))
            nn.init.zeros_(self.code_head.weight)
            nn.init.zeros_(self.code_head.bias)
        else:
            assert not ce_output, "ce_output requires use_codebook_ce"
            self.code_head = None

    def forward(self, x, t, labels, texts: list[str] | None = None, sem_token=None):
        """
        labels    : (B,) int class indices — always used for AdaLN timestep+class cond.
        texts     : list of B strings — activates semantic routing cross-attention.
        sem_token : (B, reg_hidden_size) noised DINOv2 CLS token for REG, or None.

        Always returns (x0_pred, cls0_pred, maskgit_ids_keep, code_logits).
        cls0_pred is None when REG is inactive.
        maskgit_ids_keep is None at inference or when use_maskgit=False; when not None,
        x0_pred is in patch token space (B, n_keep, patch_dim) instead of image space.
        code_logits is None unless the codebook-CE head is active (training only);
        when set it is (B, sum(fsq_levels), H, W) — per-position per-dim level logits.
        """
        p = self.patch_size
        x = rearrange(x, "b c (h p1) (w p2) -> b (h w) (c p1 p2)", p1=p, p2=p)
        x = self.patch_embed(x) + self.pos_embed
        cond = self.t_embed(t) + self.cls_embed(labels)

        # REG: prepend the noised semantic CLS token as a single extra token
        reg_active = sem_token is not None and self.sem_proj is not None
        if reg_active:
            sem_emb = self.sem_proj(sem_token).unsqueeze(1)  # (B, 1, d)
            x = torch.cat([sem_emb, x], dim=1)               # (B, 1+N, d)

        # MaskGIT: drop a random subset of patch tokens for the entire forward pass
        maskgit_ids_keep = None
        if self.use_maskgit and self.training:
            if reg_active:
                sem_tok, patch_x = x[:, :1], x[:, 1:]
            else:
                patch_x = x
            maskgit_ids_keep = TREADRouter.get_ids(
                patch_x.size(1), self.maskgit_ratio, patch_x.size(0), x.device
            )
            patch_x = TREADRouter.gather(patch_x, maskgit_ids_keep)
            x = torch.cat([sem_tok, patch_x], dim=1) if reg_active else patch_x

        # Pre-compute per-depth routed text features if routing is active
        if self.router is not None and texts is not None:
            hidden_states = self.text_encoder.encode(texts, x.device)
            contexts = [self.router.fuse(hidden_states, d) for d in range(self.depth)]
        else:
            contexts = [None] * self.depth

        tread_on       = self.use_tread and self.training
        tread_ids_keep = None
        tread_x_saved  = None

        for i, (blk, ctx) in enumerate(zip(self.blocks, contexts)):

            # TREAD route start: shrink patch sequence.
            # When reg_active the semantic token lives at position 0 and is NEVER routed —
            # tread_ids_keep indexes into patch_x (positions 1..) only, so the semantic
            # token sees every transformer block regardless of the routing schedule.
            if tread_on and i == self.tread_route_start:
                if reg_active:
                    sem_tok, patch_x = x[:, :1], x[:, 1:]
                else:
                    patch_x = x
                tread_x_saved  = patch_x                        # (B, N_patch, d) — patch tokens only
                tread_ids_keep = TREADRouter.get_ids(
                    patch_x.size(1), self.tread_selection_rate, patch_x.size(0), x.device
                )                                               # indices in [0, N_patch) — safe
                patch_x = TREADRouter.gather(patch_x, tread_ids_keep)
                x = torch.cat([sem_tok, patch_x], dim=1) if reg_active else patch_x

            if self.checkpoint_every > 0 and i % self.checkpoint_every == 0 and self.training:
                x = checkpoint(blk, x, cond, ctx, use_reentrant=False)
            else:
                x = blk(x, cond, context=ctx)

            # TREAD route end: reinsert teleported tokens
            if tread_on and i == self.tread_route_end and tread_ids_keep is not None:
                if reg_active:
                    sem_tok, patch_x = x[:, :1], x[:, 1:]
                else:
                    patch_x = x
                patch_x = TREADRouter.scatter(patch_x, tread_ids_keep, tread_x_saved)
                x = torch.cat([sem_tok, patch_x], dim=1) if reg_active else patch_x
                tread_ids_keep = None

        x = self.norm_out(x)

        if reg_active:
            # First token → semantic prediction; remaining tokens → image patches
            cls0_pred = self.sem_out(x[:, 0, :])  # (B, reg_hidden_size)
            img_out   = self.out(x[:, 1:, :])      # (B, n_keep_or_N, patch_dim)
        else:
            img_out   = self.out(x)
            cls0_pred = None

        if maskgit_ids_keep is None:
            h = w = self.img_size // p
            x0_pred = rearrange(img_out, "b (h w) (c p1 p2) -> b c (h p1) (w p2)",
                                h=h, w=w, p1=p, p2=p, c=self.channels)
        else:
            # Masked training: return patch-space predictions; caller computes loss directly
            x0_pred = img_out  # (B, n_keep, patch_dim)

        # Codebook-CE head. Computed in training (for the CE loss) and also at
        # inference when ce_output is set (so sampling can be driven by it).
        # MaskGIT excluded so the full grid is present; reg_active is incompatible
        # with the tokenizer, so x has no semantic token here.
        code_logits = None
        if self.code_head is not None and (self.training or self.ce_output) and maskgit_ids_keep is None:
            h = w = self.img_size // p
            L = sum(self.fsq_levels)
            code_logits = rearrange(self.code_head(x),
                                    "b (h w) (L p1 p2) -> b L (h p1) (w p2)",
                                    h=h, w=w, p1=p, p2=p, L=L)
            # CE-only ablation at sampling: replace the (untrained) regression
            # output with codes decoded from the argmax of the level logits.
            if self.ce_output and not self.training:
                x0_pred = self._codes_from_logits(code_logits)

        return x0_pred, cls0_pred, maskgit_ids_keep, code_logits

    def _codes_from_logits(self, code_logits):
        """argmax per-FSQ-dim level logits → normalised code values (B, n_dims, H, W).

        Inverts FSQ _scale_and_shift: code = (level - half_width) / half_width,
        matching the encoder's `codes` space (assumes identity latent_scale/shift).
        """
        codes, offset = [], 0
        for L in self.fsq_levels:
            level = code_logits[:, offset:offset + L].argmax(dim=1)  # (B, H, W)
            offset += L
            half = L // 2
            codes.append((level.float() - half) / half)
        return torch.stack(codes, dim=1)
