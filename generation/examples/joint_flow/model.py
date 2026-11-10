"""
FiLM-MLP velocity field for joint-flow, plus CVAE encoder/prior for latent task-space.

Architecture:
  t (scalar) → TimestepEmbedder → 256D  ─┐
  cond (9D)  → Linear(256)      → 256D  ─┴─ concat(512D) → cond_proj → (scale, shift) × 3

  x_t (72D) → Linear(512) ─→ FiLM → SiLU ─→
              Linear(512) ─→ FiLM → SiLU ─→
              Linear(512) ─→ FiLM → SiLU ─→
              Linear(72)                      → velocity (72D)

Reuses TimestepEmbedder / modulate patterns from examples/text/model/transformer.py.
"""

import math

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Time embedding (sinusoidal → 2-layer MLP)
# ---------------------------------------------------------------------------


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, freq_embed_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(freq_embed_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.freq_embed_size = freq_embed_size

    @staticmethod
    def _sinusoidal(t: Tensor, dim: int, max_period: float = 10000.0) -> Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t.float()[:, None] * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # (B, dim)

    def forward(self, t: Tensor) -> Tensor:
        return self.mlp(self._sinusoidal(t, self.freq_embed_size))


# ---------------------------------------------------------------------------
# FiLM (scale + shift) conditioning
# ---------------------------------------------------------------------------


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1.0 + scale) + shift


class DiTBlock(nn.Module):
    """Transformer block with timestep/condition AdaLN modulation."""

    def __init__(
        self,
        hidden_size: int,
        n_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            hidden_size,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_size),
        )
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN(
            c
        ).chunk(6, dim=-1)
        x_msa = modulate(self.norm1(x), shift_msa[:, None], scale_msa[:, None])
        attn_out = self.attn(x_msa, x_msa, x_msa, need_weights=False)[0]
        x = x + gate_msa[:, None] * attn_out
        x_mlp = modulate(self.norm2(x), shift_mlp[:, None], scale_mlp[:, None])
        x = x + gate_mlp[:, None] * self.mlp(x_mlp)
        return x


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class JointFlowMLP(nn.Module):
    """FiLM-conditioned MLP velocity field used by legacy joint-flow models."""

    def __init__(
        self,
        joint_dim: int = 69,  # 23 joints × 3 (pelvis excluded — always [0,0,0])
        cond_dim: int = 90,  # K*9, default K=10
        hidden_size: int = 512,
        n_layers: int = 3,
        t_embed_dim: int = 256,
    ):
        super().__init__()
        self.joint_dim = joint_dim

        # --- embedders ---
        self.t_embedder = TimestepEmbedder(t_embed_dim)
        self.cond_encoder = nn.Sequential(
            nn.Linear(cond_dim, t_embed_dim),
            nn.SiLU(),
            nn.Linear(t_embed_dim, t_embed_dim),
        )

        # combined condition → per-layer (scale, shift) pairs
        # one Linear produces 2 * hidden_size values per layer
        self.cond_proj = nn.Linear(t_embed_dim * 2, n_layers * 2 * hidden_size)

        # --- MLP trunk ---
        self.input_proj = nn.Linear(joint_dim, hidden_size)
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(n_layers)]
        )
        self.output_proj = nn.Linear(hidden_size, joint_dim)

        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self._init_weights()

    def _init_weights(self):
        # zero-init output projection so initial velocity ≈ 0
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x: Tensor, t: Tensor, cond: Tensor) -> Tensor:
        t_emb = self.t_embedder(t)  # (B, t_embed_dim)
        c_emb = self.cond_encoder(cond)  # (B, t_embed_dim)
        combined = torch.cat([t_emb, c_emb], dim=-1)  # (B, 2*t_embed_dim)

        # all per-layer film params at once: (B, n_layers * 2 * hidden_size)
        film_params = self.cond_proj(combined)
        # split into list of (scale, shift) per layer
        chunks = film_params.chunk(
            self.n_layers * 2, dim=-1
        )  # 2*n_layers tensors of (B, hidden)

        h = self.input_proj(x)  # (B, hidden)
        for i, layer in enumerate(self.layers):
            scale = chunks[2 * i]
            shift = chunks[2 * i + 1]
            h = modulate(h, shift, scale)
            h = torch.nn.functional.silu(layer(h))

        return self.output_proj(h)  # (B, joint_dim)


# ---------------------------------------------------------------------------
# Concat-conditioned velocity field (x15 horizon models)
# ---------------------------------------------------------------------------


class JointFlowConcatMLP(nn.Module):
    """
    Velocity field with the condition concatenated into the trunk input and
    residual hidden layers.

    Needed for large targets (x15: 760D horizon prediction): FiLM-only
    conditioning fails to ingest the condition there — the model converges
    to a near-unconditional mean (overfit check: FiLM plateaus at flow
    loss ~0.95 on 4 clips while this architecture reaches ~0.15).
    """

    def __init__(
        self,
        joint_dim: int = 69,
        cond_dim: int = 90,
        hidden_size: int = 512,
        n_layers: int = 3,
        t_embed_dim: int = 256,
    ):
        super().__init__()
        self.joint_dim = joint_dim
        self.t_embedder = TimestepEmbedder(t_embed_dim)
        self.input_proj = nn.Linear(joint_dim + cond_dim + t_embed_dim, hidden_size)
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(n_layers)]
        )
        self.output_proj = nn.Linear(hidden_size, joint_dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x: Tensor, t: Tensor, cond: Tensor) -> Tensor:
        t_emb = self.t_embedder(t)
        h = torch.nn.functional.silu(
            self.input_proj(torch.cat([x, cond, t_emb], dim=-1))
        )
        for layer in self.layers:
            h = h + torch.nn.functional.silu(layer(h))
        return self.output_proj(h)


# ---------------------------------------------------------------------------
# Transformer velocity field (x15 horizon models)
# ---------------------------------------------------------------------------


class JointFlowTransformer(nn.Module):
    """
    DiT-style velocity field over horizon frame tokens.

    The flattened flow target is reshaped as (B, H, frame_dim), projected to
    tokens, given learned horizon position embeddings, modulated by t+cond in
    each transformer block, then projected back to a flattened velocity.
    """

    def __init__(
        self,
        joint_dim: int = 760,
        cond_dim: int = 161,
        hidden_size: int = 512,
        n_layers: int = 6,
        t_embed_dim: int = 256,
        frame_dim: int = 76,
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if joint_dim % frame_dim != 0:
            raise ValueError(
                f"joint_dim={joint_dim} must be divisible by frame_dim={frame_dim}"
            )
        if hidden_size % n_heads != 0:
            raise ValueError(
                f"hidden_size={hidden_size} must be divisible by n_heads={n_heads}"
            )
        self.joint_dim = joint_dim
        self.frame_dim = frame_dim
        self.horizon = joint_dim // frame_dim

        self.x_embedder = nn.Linear(frame_dim, hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.horizon, hidden_size))
        self.t_embedder = TimestepEmbedder(t_embed_dim)
        self.cond_encoder = nn.Sequential(
            nn.Linear(cond_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.t_proj = (
            nn.Identity()
            if t_embed_dim == hidden_size
            else nn.Linear(t_embed_dim, hidden_size)
        )
        self.blocks = nn.ModuleList(
            [
                DiTBlock(hidden_size, n_heads, mlp_ratio=mlp_ratio, dropout=dropout)
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.out_mod = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))
        self.output_proj = nn.Linear(hidden_size, frame_dim)

        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.zeros_(self.out_mod[-1].weight)
        nn.init.zeros_(self.out_mod[-1].bias)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x: Tensor, t: Tensor, cond: Tensor) -> Tensor:
        x_tok = x.reshape(x.shape[0], self.horizon, self.frame_dim)
        h = self.x_embedder(x_tok) + self.pos_embed
        c = self.t_proj(self.t_embedder(t)) + self.cond_encoder(cond)
        for block in self.blocks:
            h = block(h, c)
        shift, scale = self.out_mod(c).chunk(2, dim=-1)
        h = modulate(self.norm(h), shift[:, None], scale[:, None])
        return self.output_proj(h).reshape(x.shape[0], self.joint_dim)


# ---------------------------------------------------------------------------
# History-token transformer velocity field (x16 horizon models)
# ---------------------------------------------------------------------------


class JointFlowHistoryTransformer(nn.Module):
    """
    DiT-style velocity field where the K past frames are attended over as
    sequence tokens (x16), instead of being flattened into the global FiLM
    condition as in JointFlowTransformer (x15).

    The condition vector is laid out as
        [past_frames (K*frame_dim) | task_cond (rest)].
    The past block is reshaped to K tokens and prepended to the H noised
    horizon tokens; the transformer runs full self-attention over the whole
    (K+H) sequence. Only the task-space signals (phase, heading, goal, ...)
    plus the timestep drive the adaLN modulation. Velocity is read out from the
    H horizon tokens only — the past-token outputs are discarded.

    This follows the PARC/ETH convention of keeping history as attended tokens,
    so the per-frame, per-joint temporal structure of the past is preserved
    rather than crushed through a single condition MLP.
    """

    def __init__(
        self,
        joint_dim: int = 760,
        cond_dim: int = 161,
        hidden_size: int = 512,
        n_layers: int = 6,
        t_embed_dim: int = 256,
        frame_dim: int = 76,
        past_frames: int = 2,
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if joint_dim % frame_dim != 0:
            raise ValueError(
                f"joint_dim={joint_dim} must be divisible by frame_dim={frame_dim}"
            )
        if hidden_size % n_heads != 0:
            raise ValueError(
                f"hidden_size={hidden_size} must be divisible by n_heads={n_heads}"
            )
        past_dim = past_frames * frame_dim
        task_cond_dim = cond_dim - past_dim
        if task_cond_dim < 0:
            raise ValueError(
                f"cond_dim={cond_dim} is smaller than the past block "
                f"past_frames*frame_dim={past_dim}"
            )
        self.joint_dim = joint_dim
        self.frame_dim = frame_dim
        self.past_frames = past_frames
        self.task_cond_dim = task_cond_dim
        self.horizon = joint_dim // frame_dim

        # Separate embedders: past frames are normalized with cond stats and the
        # horizon with target stats, so they live in different normalized spaces.
        self.x_embedder = nn.Linear(frame_dim, hidden_size)
        self.past_embedder = nn.Linear(frame_dim, hidden_size)
        # Absolute temporal positions spanning the past->future boundary.
        self.pos_embed = nn.Parameter(
            torch.zeros(1, past_frames + self.horizon, hidden_size)
        )
        self.t_embedder = TimestepEmbedder(t_embed_dim)
        self.t_proj = (
            nn.Identity()
            if t_embed_dim == hidden_size
            else nn.Linear(t_embed_dim, hidden_size)
        )
        self.task_encoder = (
            nn.Sequential(
                nn.Linear(task_cond_dim, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
            )
            if task_cond_dim > 0
            else None
        )
        self.blocks = nn.ModuleList(
            [
                DiTBlock(hidden_size, n_heads, mlp_ratio=mlp_ratio, dropout=dropout)
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.out_mod = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))
        self.output_proj = nn.Linear(hidden_size, frame_dim)

        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.zeros_(self.out_mod[-1].weight)
        nn.init.zeros_(self.out_mod[-1].bias)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x: Tensor, t: Tensor, cond: Tensor) -> Tensor:
        B = x.shape[0]
        past_dim = self.past_frames * self.frame_dim
        past = cond[:, :past_dim].reshape(B, self.past_frames, self.frame_dim)
        task = cond[:, past_dim:]

        past_tok = self.past_embedder(past)
        x_tok = self.x_embedder(x.reshape(B, self.horizon, self.frame_dim))
        h = torch.cat([past_tok, x_tok], dim=1) + self.pos_embed

        c = self.t_proj(self.t_embedder(t))
        if self.task_encoder is not None:
            c = c + self.task_encoder(task)
        for block in self.blocks:
            h = block(h, c)

        h = h[:, self.past_frames :]  # velocity read out from horizon tokens only
        shift, scale = self.out_mod(c).chunk(2, dim=-1)
        h = modulate(self.norm(h), shift[:, None], scale[:, None])
        return self.output_proj(h).reshape(B, self.joint_dim)


# ---------------------------------------------------------------------------
# Cross-attention history transformer velocity field (x16c horizon models)
# ---------------------------------------------------------------------------


class DiTCrossBlock(nn.Module):
    """DiT block with self-attention over the horizon, cross-attention into a
    fixed past-memory, and an MLP. All sub-layers are adaLN-modulated by the
    timestep/task condition; the cross-attention memory is never modulated."""

    def __init__(
        self,
        hidden_size: int,
        n_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.self_attn = nn.MultiheadAttention(
            hidden_size, n_heads, dropout=dropout, batch_first=True
        )
        self.norm_ca = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cross_attn = nn.MultiheadAttention(
            hidden_size, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_size),
        )
        # 7 modulation params: self-attn (shift, scale, gate), cross-attn (gate),
        # mlp (shift, scale, gate).
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 7 * hidden_size))
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x: Tensor, mem: Tensor, c: Tensor) -> Tensor:
        (
            shift_msa,
            scale_msa,
            gate_msa,
            gate_ca,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN(c).chunk(7, dim=-1)
        x_msa = modulate(self.norm1(x), shift_msa[:, None], scale_msa[:, None])
        x = x + gate_msa[:, None] * self.self_attn(
            x_msa, x_msa, x_msa, need_weights=False
        )[0]
        x_ca = self.norm_ca(x)
        x = x + gate_ca[:, None] * self.cross_attn(
            x_ca, mem, mem, need_weights=False
        )[0]
        x_mlp = modulate(self.norm2(x), shift_mlp[:, None], scale_mlp[:, None])
        x = x + gate_mlp[:, None] * self.mlp(x_mlp)
        return x


class JointFlowCrossTransformer(nn.Module):
    """
    x16c: cross-attention variant of JointFlowHistoryTransformer (x16).

    The K past frames are embedded once into read-only memory tokens and are
    never updated by the noised horizon nor modulated by the flow timestep.
    Only the H horizon tokens flow through the blocks; each block self-attends
    over the horizon, then cross-attends into the past memory. This keeps the
    history a clean, t-invariant conditioning signal (vs x16, where past tokens
    share the self-attention sequence and adaLN modulation with the horizon).

    Condition vector layout matches JointFlowHistoryTransformer:
        [past (K*frame_dim) | task_cond (rest)].
    adaLN carries timestep + task; velocity is read from all H horizon tokens.
    """

    def __init__(
        self,
        joint_dim: int = 760,
        cond_dim: int = 161,
        hidden_size: int = 512,
        n_layers: int = 6,
        t_embed_dim: int = 256,
        frame_dim: int = 76,
        past_frames: int = 2,
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if joint_dim % frame_dim != 0:
            raise ValueError(
                f"joint_dim={joint_dim} must be divisible by frame_dim={frame_dim}"
            )
        if hidden_size % n_heads != 0:
            raise ValueError(
                f"hidden_size={hidden_size} must be divisible by n_heads={n_heads}"
            )
        past_dim = past_frames * frame_dim
        task_cond_dim = cond_dim - past_dim
        if task_cond_dim < 0:
            raise ValueError(
                f"cond_dim={cond_dim} is smaller than the past block "
                f"past_frames*frame_dim={past_dim}"
            )
        self.joint_dim = joint_dim
        self.frame_dim = frame_dim
        self.past_frames = past_frames
        self.task_cond_dim = task_cond_dim
        self.horizon = joint_dim // frame_dim

        self.x_embedder = nn.Linear(frame_dim, hidden_size)
        self.past_embedder = nn.Linear(frame_dim, hidden_size)
        # Separate position embeddings for the two streams (different spaces).
        self.past_pos_embed = nn.Parameter(torch.zeros(1, past_frames, hidden_size))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.horizon, hidden_size))
        self.t_embedder = TimestepEmbedder(t_embed_dim)
        self.t_proj = (
            nn.Identity()
            if t_embed_dim == hidden_size
            else nn.Linear(t_embed_dim, hidden_size)
        )
        self.task_encoder = (
            nn.Sequential(
                nn.Linear(task_cond_dim, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
            )
            if task_cond_dim > 0
            else None
        )
        self.blocks = nn.ModuleList(
            [
                DiTCrossBlock(hidden_size, n_heads, mlp_ratio=mlp_ratio, dropout=dropout)
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.out_mod = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))
        self.output_proj = nn.Linear(hidden_size, frame_dim)

        nn.init.normal_(self.past_pos_embed, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.zeros_(self.out_mod[-1].weight)
        nn.init.zeros_(self.out_mod[-1].bias)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x: Tensor, t: Tensor, cond: Tensor) -> Tensor:
        B = x.shape[0]
        past_dim = self.past_frames * self.frame_dim
        past = cond[:, :past_dim].reshape(B, self.past_frames, self.frame_dim)
        task = cond[:, past_dim:]

        # Past memory: encoded once, never modulated by t, never updated by noise.
        mem = self.past_embedder(past) + self.past_pos_embed
        h = self.x_embedder(x.reshape(B, self.horizon, self.frame_dim)) + self.pos_embed

        c = self.t_proj(self.t_embedder(t))
        if self.task_encoder is not None:
            c = c + self.task_encoder(task)
        for block in self.blocks:
            h = block(h, mem, c)

        shift, scale = self.out_mod(c).chunk(2, dim=-1)
        h = modulate(self.norm(h), shift[:, None], scale[:, None])
        return self.output_proj(h).reshape(B, self.joint_dim)


# ---------------------------------------------------------------------------
# Full-condition cross-attention transformer velocity field (x16f horizon models)
# ---------------------------------------------------------------------------


class JointFlowFullCrossTransformer(nn.Module):
    """
    x16f: every condition is a cross-attention memory token.

    Where x16c (JointFlowCrossTransformer) cross-attends only into the K past
    frames and crushes the task-space signals (phase, heading, goal, ...) into a
    single fused adaLN vector, x16f tokenizes *all* conditions into the
    cross-attention memory bank:

        memory = [past_0 .. past_{K-1} | phase | heading | goal | speed?]

    Each task group gets its own embedder and a learned slot embedding, so the
    horizon can attend to each command source independently rather than through a
    single FiLM channel. adaLN carries the flow timestep only; every condition —
    past and task — flows in through cross-attention.

    Condition vector layout matches the other horizon models:
        [past (K*frame_dim) | task_cond (rest)].
    task_groups gives the per-group widths of the task block and must sum to
    task_cond_dim = cond_dim - K*frame_dim.
    """

    def __init__(
        self,
        joint_dim: int = 760,
        cond_dim: int = 161,
        hidden_size: int = 512,
        n_layers: int = 6,
        t_embed_dim: int = 256,
        frame_dim: int = 76,
        past_frames: int = 2,
        task_groups: tuple = (5, 2, 3),
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if joint_dim % frame_dim != 0:
            raise ValueError(
                f"joint_dim={joint_dim} must be divisible by frame_dim={frame_dim}"
            )
        if hidden_size % n_heads != 0:
            raise ValueError(
                f"hidden_size={hidden_size} must be divisible by n_heads={n_heads}"
            )
        past_dim = past_frames * frame_dim
        task_cond_dim = cond_dim - past_dim
        if task_cond_dim < 0:
            raise ValueError(
                f"cond_dim={cond_dim} is smaller than the past block "
                f"past_frames*frame_dim={past_dim}"
            )
        task_groups = tuple(int(g) for g in task_groups)
        if sum(task_groups) != task_cond_dim:
            raise ValueError(
                f"task_groups={task_groups} sum to {sum(task_groups)}, "
                f"but task_cond_dim=cond_dim-past={task_cond_dim}"
            )
        self.joint_dim = joint_dim
        self.frame_dim = frame_dim
        self.past_frames = past_frames
        self.task_groups = task_groups
        self.task_cond_dim = task_cond_dim
        self.horizon = joint_dim // frame_dim

        self.x_embedder = nn.Linear(frame_dim, hidden_size)
        self.past_embedder = nn.Linear(frame_dim, hidden_size)
        self.task_embedders = nn.ModuleList(
            [nn.Linear(g, hidden_size) for g in task_groups]
        )
        # One learned slot embedding per memory token (K past + len(task_groups)),
        # so the horizon can tell each command source apart.
        n_mem = past_frames + len(task_groups)
        self.mem_pos_embed = nn.Parameter(torch.zeros(1, n_mem, hidden_size))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.horizon, hidden_size))
        self.t_embedder = TimestepEmbedder(t_embed_dim)
        self.t_proj = (
            nn.Identity()
            if t_embed_dim == hidden_size
            else nn.Linear(t_embed_dim, hidden_size)
        )
        self.blocks = nn.ModuleList(
            [
                DiTCrossBlock(hidden_size, n_heads, mlp_ratio=mlp_ratio, dropout=dropout)
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.out_mod = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))
        self.output_proj = nn.Linear(hidden_size, frame_dim)

        nn.init.normal_(self.mem_pos_embed, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.zeros_(self.out_mod[-1].weight)
        nn.init.zeros_(self.out_mod[-1].bias)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x: Tensor, t: Tensor, cond: Tensor) -> Tensor:
        B = x.shape[0]
        past_dim = self.past_frames * self.frame_dim
        past = cond[:, :past_dim].reshape(B, self.past_frames, self.frame_dim)
        task = cond[:, past_dim:]

        # Memory bank: K past-frame tokens + one token per task group.
        mem_tokens = [self.past_embedder(past)]
        offset = 0
        for emb, g in zip(self.task_embedders, self.task_groups):
            mem_tokens.append(emb(task[:, offset : offset + g]).unsqueeze(1))
            offset += g
        mem = torch.cat(mem_tokens, dim=1) + self.mem_pos_embed

        h = self.x_embedder(x.reshape(B, self.horizon, self.frame_dim)) + self.pos_embed
        c = self.t_proj(self.t_embedder(t))
        for block in self.blocks:
            h = block(h, mem, c)

        shift, scale = self.out_mod(c).chunk(2, dim=-1)
        h = modulate(self.norm(h), shift[:, None], scale[:, None])
        return self.output_proj(h).reshape(B, self.joint_dim)


# ---------------------------------------------------------------------------
# CVAE encoder and prior for latent task-space conditioning
# ---------------------------------------------------------------------------


class CVAEEncoder(nn.Module):
    """
    Training-time encoder: (cond, joints) → (μ, log_var).
    Sees both task-space condition and target joints so it can produce
    an informative latent z that captures what the flow model needs.
    """

    def __init__(
        self, cond_dim: int, joint_dim: int, latent_dim: int, hidden: int = 256
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim + joint_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.mu_head = nn.Linear(hidden, latent_dim)
        self.logvar_head = nn.Linear(hidden, latent_dim)

    def forward(self, cond: Tensor, joints: Tensor):
        h = self.net(torch.cat([cond, joints], dim=-1))
        return self.mu_head(h), self.logvar_head(h)


class CVAEPrior(nn.Module):
    """
    Inference-time prior: cond → (μ, log_var).
    Trained to match the encoder distribution — at inference the planner
    only has task-space, so this is the only path available.
    """

    def __init__(self, cond_dim: int, latent_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.mu_head = nn.Linear(hidden, latent_dim)
        self.logvar_head = nn.Linear(hidden, latent_dim)

    def forward(self, cond: Tensor):
        h = self.net(cond)
        return self.mu_head(h), self.logvar_head(h)


def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
    std = torch.exp(0.5 * logvar)
    return mu + torch.randn_like(std) * std


def kl_divergence(
    mu_enc: Tensor, logvar_enc: Tensor, mu_prior: Tensor, logvar_prior: Tensor
) -> Tensor:
    """KL(N(μ_enc, σ_enc²) ‖ N(μ_prior, σ_prior²)), mean over batch."""
    return (
        0.5
        * (
            logvar_prior
            - logvar_enc
            + (logvar_enc.exp() + (mu_enc - mu_prior).pow(2)) / logvar_prior.exp()
            - 1.0
        )
        .sum(dim=-1)
        .mean()
    )
