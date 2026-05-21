import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


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


class DiTBlock(nn.Module):
    def __init__(self, d: int, heads: int, cond_d: int, use_cross_attn: bool = False):
        super().__init__()
        self.norm1 = RMSNorm(d)
        self.norm2 = RMSNorm(d)
        self.attn  = Attention(d, heads)
        self.ffn   = SwiGLU(d)
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
    ):
        super().__init__()
        self.img_size   = img_size
        self.patch_size = patch_size
        self.channels   = channels
        self.depth      = depth
        self.use_semantic_routing = use_semantic_routing
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

        self.blocks   = nn.ModuleList([
            DiTBlock(d, heads, d, use_cross_attn=use_semantic_routing)
            for _ in range(depth)
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

    def forward(self, x, t, labels, texts: list[str] | None = None):
        """
        labels : (B,) int class indices — always used for AdaLN timestep+class cond.
        texts  : list of B strings — activates semantic routing cross-attention.
                 If None, cross-attention is skipped (falls back to class-only).
        """
        p = self.patch_size
        x = rearrange(x, "b c (h p1) (w p2) -> b (h w) (c p1 p2)", p1=p, p2=p)
        x = self.patch_embed(x) + self.pos_embed
        cond = self.t_embed(t) + self.cls_embed(labels)

        # Pre-compute per-depth routed text features if routing is active
        if self.router is not None and texts is not None:
            hidden_states = self.text_encoder.encode(texts, x.device)
            contexts = [self.router.fuse(hidden_states, d) for d in range(self.depth)]
        else:
            contexts = [None] * self.depth

        for blk, ctx in zip(self.blocks, contexts):
            x = blk(x, cond, context=ctx)

        x = self.out(self.norm_out(x))
        h = w = self.img_size // p
        return rearrange(x, "b (h w) (c p1 p2) -> b c (h p1) (w p2)",
                         h=h, w=w, p1=p, p2=p, c=self.channels)
