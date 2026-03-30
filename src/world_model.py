"""JEPA-based World Model for Kinetix — JAX/Flax NNX port of le-wm.

Architecture:
  - VisionEncoder: ViT-like encoder (patch embed + transformer blocks)
  - ARPredictor: Autoregressive predictor with AdaLN conditioning
  - ActionEmbedder: Conv1d + MLP action encoder
  - SIGReg: Sketch Isotropic Gaussian Regularizer
  - JEPA: Full world model combining all components

References:
  - LeWorldModel (le-wm): https://github.com/...
  - "A Stable End-to-End JEPA for Pixel-Based World Models" (Maes et al.)
"""

import dataclasses
import functools
from typing import Sequence

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

@dataclasses.dataclass(frozen=True)
class WorldModelConfig:
    # Vision encoder
    img_size: int = 64
    patch_size: int = 8
    encoder_dim: int = 192
    encoder_depth: int = 6
    encoder_heads: int = 3

    # Predictor
    pred_depth: int = 6
    pred_heads: int = 8
    pred_mlp_dim: int = 512
    pred_dim_head: int = 64
    pred_dropout: float = 0.1

    # Embedding
    embed_dim: int = 192
    proj_hidden_dim: int = 512

    # Action encoder
    action_dim: int = 6  # will be set from env

    # World model
    history_size: int = 3
    num_preds: int = 1

    # Decoder
    decoder_depth: int = 4
    decoder_dim: int = 192
    recon_weight: float = 1.0  # pixel reconstruction loss weight

    # Training
    sigreg_weight: float = 0.09
    sigreg_knots: int = 17
    sigreg_num_proj: int = 1024


# ─────────────────────────────────────────────
# Vision Encoder (ViT)
# ─────────────────────────────────────────────

class PatchEmbed(nnx.Module):
    """Convert image to patch embeddings via convolution."""

    def __init__(self, img_size: int, patch_size: int, in_channels: int, embed_dim: int, *, rngs: nnx.Rngs):
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.proj = nnx.Conv(
            in_features=in_channels,
            out_features=embed_dim,
            kernel_size=(patch_size, patch_size),
            strides=(patch_size, patch_size),
            padding="VALID",
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        # x: (B, H, W, C)
        x = self.proj(x)  # (B, H', W', embed_dim)
        b = x.shape[0]
        x = x.reshape(b, -1, x.shape[-1])  # (B, num_patches, embed_dim)
        return x


class TransformerBlock(nnx.Module):
    """Standard pre-norm transformer block."""

    def __init__(self, dim: int, heads: int, dim_head: int, mlp_dim: int, dropout: float = 0.0, *, rngs: nnx.Rngs):
        self.norm1 = nnx.LayerNorm(dim, rngs=rngs)
        self.norm2 = nnx.LayerNorm(dim, rngs=rngs)

        inner_dim = heads * dim_head
        self.to_qkv = nnx.Linear(dim, inner_dim * 3, use_bias=False, rngs=rngs)
        self.to_out = nnx.Linear(inner_dim, dim, rngs=rngs)
        self.heads = heads
        self.dim_head = dim_head

        self.mlp_norm = nnx.LayerNorm(dim, rngs=rngs)
        self.mlp_in = nnx.Linear(dim, mlp_dim, rngs=rngs)
        self.mlp_out = nnx.Linear(mlp_dim, dim, rngs=rngs)
        self.dropout = nnx.Dropout(rate=dropout, rngs=rngs)

    def __call__(self, x: jax.Array, deterministic: bool = True) -> jax.Array:
        # Self-attention
        residual = x
        x_norm = self.norm1(x)
        qkv = self.to_qkv(x_norm)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        b, t = q.shape[0], q.shape[1]
        q = q.reshape(b, t, self.heads, self.dim_head).transpose(0, 2, 1, 3)
        k = k.reshape(b, t, self.heads, self.dim_head).transpose(0, 2, 1, 3)
        v = v.reshape(b, t, self.heads, self.dim_head).transpose(0, 2, 1, 3)

        scale = self.dim_head ** -0.5
        attn = jnp.matmul(q, k.transpose(0, 1, 3, 2)) * scale
        attn = jax.nn.softmax(attn, axis=-1)
        out = jnp.matmul(attn, v)
        out = out.transpose(0, 2, 1, 3).reshape(b, t, -1)
        out = self.to_out(out)
        x = residual + out

        # FFN
        residual = x
        x_norm = self.norm2(x)
        x_mlp = self.mlp_in(x_norm)
        x_mlp = nnx.gelu(x_mlp)
        x_mlp = self.mlp_out(x_mlp)
        x = residual + x_mlp
        return x


class VisionEncoder(nnx.Module):
    """ViT-style vision encoder that outputs a CLS token embedding."""

    def __init__(self, config: WorldModelConfig, *, rngs: nnx.Rngs):
        self.patch_embed = PatchEmbed(
            config.img_size, config.patch_size, 3, config.encoder_dim, rngs=rngs
        )
        num_patches = self.patch_embed.num_patches

        # CLS token
        self.cls_token = nnx.Param(jax.random.normal(rngs.params(), (1, 1, config.encoder_dim)) * 0.02)
        # Positional embedding (CLS + patches)
        self.pos_embed = nnx.Param(
            jax.random.normal(rngs.params(), (1, num_patches + 1, config.encoder_dim)) * 0.02
        )

        self.blocks = [
            TransformerBlock(
                config.encoder_dim,
                config.encoder_heads,
                config.encoder_dim // config.encoder_heads,
                config.encoder_dim * 4,
                rngs=rngs,
            )
            for _ in range(config.encoder_depth)
        ]
        self.norm = nnx.LayerNorm(config.encoder_dim, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        x: (B, H, W, C) float32 pixels [0, 1]
        Returns: (B, encoder_dim) CLS token embedding
        """
        b = x.shape[0]
        patches = self.patch_embed(x)  # (B, num_patches, dim)

        cls_tokens = jnp.broadcast_to(self.cls_token.value, (b, 1, patches.shape[-1]))
        x = jnp.concatenate([cls_tokens, patches], axis=1)
        x = x + self.pos_embed.value[:, :x.shape[1]]

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return x[:, 0]  # CLS token


# ─────────────────────────────────────────────
# Conditional Transformer (ARPredictor)
# ─────────────────────────────────────────────

class ConditionalBlock(nnx.Module):
    """Transformer block with AdaLN-zero conditioning (action-conditioned)."""

    def __init__(self, dim: int, heads: int, dim_head: int, mlp_dim: int, dropout: float = 0.0, *, rngs: nnx.Rngs):
        # Attention
        inner_dim = heads * dim_head
        self.norm1 = nnx.LayerNorm(dim, use_scale=False, use_bias=False, rngs=rngs)
        self.to_qkv = nnx.Linear(dim, inner_dim * 3, use_bias=False, rngs=rngs)
        self.to_out = nnx.Linear(inner_dim, dim, rngs=rngs)
        self.heads = heads
        self.dim_head = dim_head

        # FFN
        self.norm2 = nnx.LayerNorm(dim, use_scale=False, use_bias=False, rngs=rngs)
        self.mlp_in = nnx.Linear(dim, mlp_dim, rngs=rngs)
        self.mlp_out = nnx.Linear(mlp_dim, dim, rngs=rngs)

        # AdaLN-zero: 6 * dim (shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp)
        self.adaln = nnx.Linear(dim, 6 * dim, kernel_init=nnx.initializers.zeros_init(), rngs=rngs)

    def __call__(self, x: jax.Array, c: jax.Array, causal: bool = True) -> jax.Array:
        """
        x: (B, T, D) input embeddings
        c: (B, T, D) conditioning (action embeddings)
        """
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = jnp.split(
            nnx.silu(c) if c.shape[-1] != x.shape[-1] else self.adaln(nnx.silu(c)), 6, axis=-1
        )
        # Actually always use adaln
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = jnp.split(
            self.adaln(nnx.silu(c)), 6, axis=-1
        )

        # Attention with AdaLN modulation
        b, t = x.shape[0], x.shape[1]
        x_mod = self.norm1(x) * (1 + scale_a) + shift_a
        qkv = self.to_qkv(x_mod)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        q = q.reshape(b, t, self.heads, self.dim_head).transpose(0, 2, 1, 3)
        k = k.reshape(b, t, self.heads, self.dim_head).transpose(0, 2, 1, 3)
        v = v.reshape(b, t, self.heads, self.dim_head).transpose(0, 2, 1, 3)

        scale = self.dim_head ** -0.5
        attn = jnp.matmul(q, k.transpose(0, 1, 3, 2)) * scale

        if causal:
            mask = jnp.tril(jnp.ones((t, t)))
            attn = jnp.where(mask[None, None, :, :], attn, -1e9)

        attn = jax.nn.softmax(attn, axis=-1)
        out = jnp.matmul(attn, v)
        out = out.transpose(0, 2, 1, 3).reshape(b, t, -1)
        out = self.to_out(out)
        x = x + gate_a * out

        # FFN with AdaLN modulation
        x_mod = self.norm2(x) * (1 + scale_m) + shift_m
        mlp_out = self.mlp_in(x_mod)
        mlp_out = nnx.gelu(mlp_out)
        mlp_out = self.mlp_out(mlp_out)
        x = x + gate_m * mlp_out

        return x


class ARPredictor(nnx.Module):
    """Autoregressive predictor for next-state embedding prediction."""

    def __init__(self, config: WorldModelConfig, *, rngs: nnx.Rngs):
        dim = config.embed_dim
        hidden_dim = config.encoder_dim  # match le-wm: hidden_dim = encoder hidden_size

        self.input_proj = nnx.Linear(dim, hidden_dim, rngs=rngs) if dim != hidden_dim else None
        self.cond_proj = nnx.Linear(dim, hidden_dim, rngs=rngs) if dim != hidden_dim else None

        self.pos_embed = nnx.Param(
            jax.random.normal(rngs.params(), (1, config.history_size, hidden_dim)) * 0.02
        )

        self.blocks = [
            ConditionalBlock(
                hidden_dim,
                config.pred_heads,
                config.pred_dim_head,
                config.pred_mlp_dim,
                config.pred_dropout,
                rngs=rngs,
            )
            for _ in range(config.pred_depth)
        ]
        self.norm = nnx.LayerNorm(hidden_dim, rngs=rngs)
        self.output_proj = nnx.Linear(hidden_dim, dim, rngs=rngs) if hidden_dim != dim else None

    def __call__(self, x: jax.Array, c: jax.Array) -> jax.Array:
        """
        x: (B, T, embed_dim) state embeddings
        c: (B, T, embed_dim) action embeddings
        Returns: (B, T, embed_dim) predicted next-state embeddings
        """
        t = x.shape[1]
        if self.input_proj is not None:
            x = self.input_proj(x)
        if self.cond_proj is not None:
            c = self.cond_proj(c)

        x = x + self.pos_embed.value[:, :t]

        for block in self.blocks:
            x = block(x, c, causal=True)

        x = self.norm(x)

        if self.output_proj is not None:
            x = self.output_proj(x)
        return x


# ─────────────────────────────────────────────
# Action Encoder
# ─────────────────────────────────────────────

class ActionEmbedder(nnx.Module):
    """Encode action sequences into embeddings (Conv1d + MLP)."""

    def __init__(self, action_dim: int, embed_dim: int, mlp_scale: int = 4, *, rngs: nnx.Rngs):
        # Conv1d over time dimension: kernel_size=1 for pointwise
        self.conv = nnx.Conv(
            in_features=action_dim,
            out_features=action_dim,
            kernel_size=(1,),
            strides=(1,),
            padding="SAME",
            rngs=rngs,
        )
        self.mlp = nnx.Sequential(
            nnx.Linear(action_dim, mlp_scale * embed_dim, rngs=rngs),
            nnx.silu,
            nnx.Linear(mlp_scale * embed_dim, embed_dim, rngs=rngs),
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        x: (B, T, action_dim)
        Returns: (B, T, embed_dim)
        """
        # Conv1d over T: Flax Conv expects (B, T, C)
        x = self.conv(x)
        x = self.mlp(x)
        return x


# ─────────────────────────────────────────────
# Projection MLPs
# ─────────────────────────────────────────────

class ProjectionMLP(nnx.Module):
    """MLP for projecting encoder/predictor outputs to embedding space."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, use_batch_norm: bool = True, *, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(input_dim, hidden_dim, rngs=rngs)
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.bn = nnx.BatchNorm(hidden_dim, rngs=rngs)
        else:
            self.ln = nnx.LayerNorm(hidden_dim, rngs=rngs)
        self.linear2 = nnx.Linear(hidden_dim, output_dim, rngs=rngs)

    def __call__(self, x: jax.Array, use_running_average: bool = True) -> jax.Array:
        """x: (B, D)"""
        x = self.linear1(x)
        if self.use_batch_norm:
            x = self.bn(x, use_running_average=use_running_average)
        else:
            x = self.ln(x)
        x = nnx.gelu(x)
        x = self.linear2(x)
        return x


# ─────────────────────────────────────────────
# SIGReg (Sketch Isotropic Gaussian Regularizer)
# ─────────────────────────────────────────────

def sigreg_loss(emb: jax.Array, knots: int = 17, num_proj: int = 1024, rng: jax.Array = None) -> jax.Array:
    """Compute SIGReg loss to prevent representation collapse.

    Args:
        emb: (T, B, D) embeddings (time-first)
        knots: number of evaluation points for characteristic function
        num_proj: number of random projections
        rng: PRNG key for random projections

    Returns:
        scalar loss value
    """
    t = jnp.linspace(0, 3, knots)
    dt = 3.0 / (knots - 1)
    weights = jnp.full((knots,), 2 * dt)
    weights = weights.at[0].set(dt)
    weights = weights.at[-1].set(dt)
    phi = jnp.exp(-t ** 2 / 2.0)
    weights = weights * phi

    # Random projections
    if rng is None:
        rng = jax.random.key(0)
    A = jax.random.normal(rng, (emb.shape[-1], num_proj))
    A = A / jnp.linalg.norm(A, axis=0, keepdims=True)

    # Compute Epps-Pulley statistic
    # emb: (T, B, D), A: (D, num_proj)
    proj = jnp.matmul(emb, A)  # (T, B, num_proj)
    x_t = proj[..., None] * t  # (T, B, num_proj, knots)

    cos_mean = jnp.cos(x_t).mean(axis=1)  # (T, num_proj, knots) - mean over B
    sin_mean = jnp.sin(x_t).mean(axis=1)  # (T, num_proj, knots)

    err = (cos_mean - phi) ** 2 + sin_mean ** 2  # (T, num_proj, knots)
    statistic = jnp.matmul(err, weights) * emb.shape[1]  # (T, num_proj) * B

    return statistic.mean()


# ─────────────────────────────────────────────
# Pixel Decoder (embedding → pixels)
# ─────────────────────────────────────────────

class PixelDecoder(nnx.Module):
    """Decode latent embeddings back to pixel images.

    Architecture: MLP → reshape to patches → transposed conv (upscale) → pixels.
    """

    def __init__(self, config: WorldModelConfig, *, rngs: nnx.Rngs):
        num_patches_per_side = config.img_size // config.patch_size
        num_patches = num_patches_per_side ** 2
        self.num_patches_per_side = num_patches_per_side
        self.patch_size = config.patch_size

        # Project embedding to patch tokens
        self.proj = nnx.Linear(config.embed_dim, num_patches * config.decoder_dim, rngs=rngs)

        # Transformer blocks to refine patch tokens
        self.blocks = [
            TransformerBlock(
                config.decoder_dim,
                config.encoder_heads,
                config.decoder_dim // config.encoder_heads,
                config.decoder_dim * 4,
                rngs=rngs,
            )
            for _ in range(config.decoder_depth)
        ]
        self.norm = nnx.LayerNorm(config.decoder_dim, rngs=rngs)

        # Project each patch token to patch pixels
        self.pixel_proj = nnx.Linear(
            config.decoder_dim,
            config.patch_size * config.patch_size * 3,
            rngs=rngs,
        )

    def __call__(self, emb: jax.Array) -> jax.Array:
        """
        emb: (B, embed_dim)
        Returns: (B, H, W, 3) float32 [0, 1]
        """
        b = emb.shape[0]
        ps = self.patch_size
        nps = self.num_patches_per_side

        # Project to patch tokens
        x = self.proj(emb)  # (B, num_patches * decoder_dim)
        x = x.reshape(b, nps * nps, -1)  # (B, num_patches, decoder_dim)

        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        # Each patch token → patch pixels
        x = self.pixel_proj(x)  # (B, num_patches, ps*ps*3)
        x = x.reshape(b, nps, nps, ps, ps, 3)
        x = x.transpose(0, 1, 3, 2, 4, 5)  # (B, nps, ps, nps, ps, 3)
        x = x.reshape(b, nps * ps, nps * ps, 3)  # (B, H, W, 3)
        x = jax.nn.sigmoid(x)  # [0, 1]
        return x


# ─────────────────────────────────────────────
# JEPA World Model
# ─────────────────────────────────────────────

class JEPA(nnx.Module):
    """Joint-Embedding Predictive Architecture world model.

    Given pixel observations and actions, learns to predict future state
    embeddings in latent space.
    """

    def __init__(self, config: WorldModelConfig, *, rngs: nnx.Rngs):
        self.config = config

        self.encoder = VisionEncoder(config, rngs=rngs)
        self.predictor = ARPredictor(config, rngs=rngs)
        self.action_encoder = ActionEmbedder(config.action_dim, config.embed_dim, rngs=rngs)
        self.projector = ProjectionMLP(
            config.encoder_dim, config.proj_hidden_dim, config.embed_dim, rngs=rngs
        )
        self.pred_proj = ProjectionMLP(
            config.encoder_dim if config.encoder_dim != config.embed_dim else config.embed_dim,
            config.proj_hidden_dim,
            config.embed_dim,
            rngs=rngs,
        )
        self.decoder = PixelDecoder(config, rngs=rngs)

    def encode(self, pixels: jax.Array, actions: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Encode pixel observations and actions.

        Args:
            pixels: (B, T, H, W, C) float32 [0, 1]
            actions: (B, T, action_dim) float32

        Returns:
            emb: (B, T, embed_dim) state embeddings
            act_emb: (B, T, embed_dim) action embeddings
        """
        b, t = pixels.shape[:2]

        # Flatten batch and time for encoding
        flat_pixels = pixels.reshape(b * t, *pixels.shape[2:])
        flat_enc = self.encoder(flat_pixels)  # (B*T, encoder_dim)
        flat_emb = self.projector(flat_enc, use_running_average=True)  # (B*T, embed_dim)
        emb = flat_emb.reshape(b, t, -1)  # (B, T, embed_dim)

        act_emb = self.action_encoder(actions)  # (B, T, embed_dim)

        return emb, act_emb

    def decode(self, emb: jax.Array) -> jax.Array:
        """Decode embeddings to pixel images.

        Args:
            emb: (B, T, embed_dim) or (B, embed_dim)

        Returns:
            pixels: same leading dims + (H, W, 3) float32 [0, 1]
        """
        if emb.ndim == 3:
            b, t = emb.shape[:2]
            flat = emb.reshape(b * t, -1)
            decoded = self.decoder(flat)
            return decoded.reshape(b, t, *decoded.shape[1:])
        return self.decoder(emb)

    def predict(self, emb: jax.Array, act_emb: jax.Array) -> jax.Array:
        """Predict next-state embeddings.

        Args:
            emb: (B, T, embed_dim) context embeddings
            act_emb: (B, T, embed_dim) action embeddings

        Returns:
            pred_emb: (B, T, embed_dim) predicted embeddings
        """
        b, t = emb.shape[:2]
        preds = self.predictor(emb, act_emb)  # (B, T, hidden_dim or embed_dim)
        # Project predictions
        flat_preds = preds.reshape(b * t, -1)
        flat_proj = self.pred_proj(flat_preds, use_running_average=True)
        return flat_proj.reshape(b, t, -1)

    def loss(
        self,
        rng: jax.Array,
        pixels: jax.Array,
        actions: jax.Array,
    ) -> tuple[jax.Array, dict]:
        """Compute world model training loss.

        Args:
            rng: PRNG key
            pixels: (B, T, H, W, C) float32 [0, 1]
            actions: (B, T, action_dim) float32

        Returns:
            total_loss: scalar
            info: dict of individual losses
        """
        cfg = self.config
        ctx_len = cfg.history_size
        n_preds = cfg.num_preds

        # Encode all frames
        emb, act_emb = self.encode(pixels, actions)

        # Context and target splits
        ctx_emb = emb[:, :ctx_len]
        ctx_act = act_emb[:, :ctx_len]
        tgt_emb = emb[:, n_preds:]  # shifted ground truth

        # Predict
        pred_emb = self.predict(ctx_emb, ctx_act)

        # Prediction loss (MSE)
        pred_loss = jnp.mean((pred_emb - jax.lax.stop_gradient(tgt_emb)) ** 2)

        # SIGReg loss
        rng, sigreg_rng = jax.random.split(rng)
        sig_loss = sigreg_loss(
            emb.transpose(1, 0, 2),  # (T, B, D)
            knots=cfg.sigreg_knots,
            num_proj=cfg.sigreg_num_proj,
            rng=sigreg_rng,
        )

        # Reconstruction loss: decode all embeddings back to pixels
        recon_pixels = self.decode(emb)  # (B, T, H, W, 3)
        recon_loss = jnp.mean((recon_pixels - pixels) ** 2)

        total_loss = pred_loss + cfg.sigreg_weight * sig_loss + cfg.recon_weight * recon_loss

        info = {
            "pred_loss": pred_loss,
            "sigreg_loss": sig_loss,
            "recon_loss": recon_loss,
            "total_loss": total_loss,
        }
        return total_loss, info

    def rollout(
        self,
        pixels: jax.Array,
        actions: jax.Array,
        future_actions: jax.Array,
    ) -> jax.Array:
        """Autoregressive rollout for planning.

        Args:
            pixels: (B, T, H, W, C) initial context frames
            actions: (B, T, action_dim) actions for context frames
            future_actions: (B, N, action_dim) future actions to predict

        Returns:
            all_emb: (B, T+N+1, embed_dim) all embeddings (context + predicted)
        """
        cfg = self.config
        hs = cfg.history_size

        # Encode initial context
        emb, _ = self.encode(pixels, actions)
        act_emb = self.action_encoder(actions)

        # Autoregressive rollout
        all_act = jnp.concatenate([actions, future_actions], axis=1)
        all_act_emb = self.action_encoder(all_act)

        n_steps = future_actions.shape[1]
        current_emb = emb

        for t in range(n_steps):
            # Use last `hs` embeddings
            ctx = current_emb[:, -hs:]
            act_ctx = all_act_emb[:, current_emb.shape[1] - hs : current_emb.shape[1]]
            pred = self.predict(ctx, act_ctx)[:, -1:]  # (B, 1, D)
            current_emb = jnp.concatenate([current_emb, pred], axis=1)

        # Final prediction
        ctx = current_emb[:, -hs:]
        act_ctx = all_act_emb[:, current_emb.shape[1] - hs : current_emb.shape[1]]
        pred = self.predict(ctx, act_ctx)[:, -1:]
        current_emb = jnp.concatenate([current_emb, pred], axis=1)

        return current_emb

    def get_goal_cost(
        self,
        pixels: jax.Array,
        actions: jax.Array,
        goal_pixels: jax.Array,
        action_candidates: jax.Array,
    ) -> jax.Array:
        """Compute planning cost for action candidates.

        Args:
            pixels: (B, T, H, W, C) initial context
            actions: (B, T, action_dim) context actions
            goal_pixels: (B, 1, H, W, C) goal frame
            action_candidates: (B, S, N, action_dim) candidate action sequences

        Returns:
            costs: (B, S) cost for each candidate
        """
        b, s, n = action_candidates.shape[:3]

        # Encode goal
        goal_flat = goal_pixels.reshape(b, 1, *goal_pixels.shape[2:])
        goal_emb, _ = self.encode(
            goal_flat,
            jnp.zeros((b, 1, self.config.action_dim)),
        )  # (B, 1, D)

        # Evaluate each candidate (vmapped over S)
        def eval_candidate(future_actions):
            # future_actions: (B, N, action_dim)
            all_emb = self.rollout(pixels, actions, future_actions)
            final_emb = all_emb[:, -1:]  # (B, 1, D)
            cost = jnp.mean((final_emb - goal_emb) ** 2, axis=-1).squeeze(-1)  # (B,)
            return cost

        # Reshape candidates for vmap
        candidates_t = action_candidates.transpose(1, 0, 2, 3)  # (S, B, N, D)
        costs = jax.vmap(eval_candidate)(candidates_t)  # (S, B)
        return costs.T  # (B, S)
