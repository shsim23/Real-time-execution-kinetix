"""Visualize world model: GT vs reconstructed vs predicted frames + latent space.

Creates for each example:
  1. Side-by-side video (gif): GT / Recon / Predicted frames
  2. Static comparison image (png)
  3. Latent space visualization (PCA of GT vs predicted embeddings)

Usage:
    uv run src/visualize_world_model.py \
        --checkpoint-dir ./logs-wm/<run>/29 \
        --data-path ./data-pixel/worlds_l_grasp_easy.npz
"""

import dataclasses
import pathlib
import pickle

import flax.nnx as nnx
import imageio
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import tyro

import world_model as wm


@dataclasses.dataclass
class Config:
    checkpoint_dir: str
    data_path: str
    output_dir: str = "wm_vis"
    num_examples: int = 8
    rollout_steps: int = 30
    seed: int = 42


def load_model(checkpoint_dir: str):
    ckpt_dir = pathlib.Path(checkpoint_dir)
    with (ckpt_dir / "config.pkl").open("rb") as f:
        model_config = pickle.load(f)
    with (ckpt_dir / "world_model.pkl").open("rb") as f:
        state_dict = pickle.load(f)

    model = wm.JEPA(model_config, rngs=nnx.Rngs(jax.random.key(0)))
    graphdef, state = nnx.split(model)
    state.replace_by_pure_dict(state_dict)
    model = nnx.merge(graphdef, state)
    return model, model_config


def to_uint8(x):
    return np.clip(np.array(x) * 255, 0, 255).astype(np.uint8)


def add_border(frame, color, pad=2):
    h, w = frame.shape[:2]
    bordered = np.full((h + 2 * pad, w + 2 * pad, 3), color, dtype=np.uint8)
    bordered[pad:pad + h, pad:pad + w] = frame
    return bordered


def find_valid_window(rng, done, total_len, num_envs, num_steps):
    """Find a window with no episode boundary."""
    for _ in range(2000):
        rng, k1, k2 = jax.random.split(rng, 3)
        env_idx = int(jax.random.randint(k1, (), 0, num_envs))
        start = int(jax.random.randint(k2, (), 0, max(1, num_steps - total_len)))
        if not done[start:start + total_len, env_idx].any():
            return rng, start, env_idx
    return rng, None, None


def make_comparison_video(gt_frames, recon_frames, pred_frames, hs):
    """Build side-by-side gif frames: GT / Recon / Predicted."""
    video = []
    n = len(gt_frames)
    for t in range(n):
        if t < hs:
            color_gt = [0, 180, 0]       # green = context
            color_pred = [0, 180, 0]
        else:
            color_gt = [180, 180, 180]    # gray = GT future
            color_pred = [220, 120, 0]    # orange = WM predicted
        color_recon = [0, 100, 220]       # blue = reconstruction

        row_gt = add_border(gt_frames[t], color_gt)
        row_rc = add_border(recon_frames[t], color_recon)
        row_pd = add_border(pred_frames[t], color_pred)

        frame = np.concatenate([row_gt, row_rc, row_pd], axis=0)
        video.append(frame)
    return video


def plot_latent_pca(gt_emb, pred_emb, hs, save_path):
    """PCA visualization of GT vs predicted embeddings over time."""
    gt = np.array(gt_emb)      # (T, D)
    pred = np.array(pred_emb)  # (T', D)

    # Fit PCA on all embeddings together
    all_emb = np.concatenate([gt, pred], axis=0)
    pca = PCA(n_components=2)
    all_2d = pca.fit_transform(all_emb)
    gt_2d = all_2d[:len(gt)]
    pred_2d = all_2d[len(gt):]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: trajectory in PCA space
    ax = axes[0]
    # GT context
    ax.plot(gt_2d[:hs, 0], gt_2d[:hs, 1], 'g-o', markersize=4, label='GT context', linewidth=2)
    # GT future
    ax.plot(gt_2d[hs:, 0], gt_2d[hs:, 1], 'k-o', markersize=3, label='GT future', linewidth=1.5)
    # Predicted future (skip context part)
    pred_future = pred_2d[hs:]
    ax.plot(pred_future[:, 0], pred_future[:, 1], '-o', color='orange',
            markersize=3, label='WM predicted', linewidth=1.5)

    # Mark start/end
    ax.scatter(*gt_2d[0], c='green', s=100, zorder=5, marker='s', edgecolors='black', label='start')
    ax.scatter(*gt_2d[-1], c='black', s=100, zorder=5, marker='^', edgecolors='black', label='GT end')
    if len(pred_future) > 0:
        ax.scatter(*pred_future[-1], c='orange', s=100, zorder=5, marker='^',
                   edgecolors='black', label='pred end')

    ax.set_title(f'Latent Trajectory (PCA, var={pca.explained_variance_ratio_.sum():.1%})')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 2: per-step MSE over time
    ax2 = axes[1]
    # Align lengths
    min_len = min(len(gt) - hs, len(pred) - hs)
    if min_len > 0:
        step_mse = np.mean((gt[hs:hs + min_len] - np.array(pred_emb)[hs:hs + min_len]) ** 2, axis=-1)
        steps = np.arange(min_len)
        ax2.bar(steps, step_mse, color='orange', alpha=0.7, label='Embedding MSE')
        ax2.axhline(y=step_mse.mean(), color='red', linestyle='--', label=f'mean={step_mse.mean():.4f}')
        ax2.set_xlabel('Future step')
        ax2.set_ylabel('MSE')
        ax2.set_title('Prediction Error Over Time')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()


def main(config: Config):
    print("Loading model...")
    model, model_config = load_model(config.checkpoint_dir)
    hs = model_config.history_size

    print("Loading data...")
    data = np.load(config.data_path)
    pixels = data["pixels"]
    actions = data["action"]
    done = data["done"]

    num_steps, num_envs = pixels.shape[:2]
    total_len = hs + config.rollout_steps
    print(f"Data: {num_steps} steps x {num_envs} envs, context={hs}, rollout={config.rollout_steps}")

    output_dir = pathlib.Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = jax.random.key(config.seed)
    all_emb_mse = []
    all_pixel_mse = []

    # Collect all embeddings for a global PCA plot
    all_gt_embs = []
    all_pred_embs = []

    for ex in range(config.num_examples):
        rng, start, env_idx = find_valid_window(rng, done, total_len, num_envs, num_steps)
        if start is None:
            print(f"  Example {ex}: no valid window, skipping")
            continue

        # Extract data
        win_px = pixels[start:start + total_len, env_idx]       # (T, H, W, 3) uint8
        win_act = actions[start:start + total_len, env_idx]     # (T, action_dim)

        ctx_px = jnp.array(win_px[:hs][None] / 255.0)          # (1, hs, H, W, 3)
        ctx_act = jnp.array(win_act[:hs][None])                 # (1, hs, A)
        fut_act = jnp.array(win_act[hs:][None])                 # (1, rollout, A)
        all_px = jnp.array(win_px[None] / 255.0)                # (1, T, H, W, 3)
        all_act = jnp.array(win_act[None])                      # (1, T, A)

        # --- Encode GT ---
        gt_emb, _ = model.encode(all_px, all_act)               # (1, T, D)

        # --- Reconstruct GT (encoder → decoder quality) ---
        recon_px = model.decode(gt_emb)                          # (1, T, H, W, 3)

        # --- WM rollout (predict future from context + actions) ---
        pred_emb = model.rollout(ctx_px, ctx_act, fut_act)      # (1, hs+rollout+1, D)

        # --- Decode predicted embeddings ---
        pred_px = model.decode(pred_emb)                         # (1, hs+rollout+1, H, W, 3)

        # Convert to numpy uint8
        gt_np = win_px                                            # (T, H, W, 3)
        recon_np = to_uint8(recon_px[0])                          # (T, H, W, 3)
        pred_np = to_uint8(pred_px[0])                            # (hs+rollout+1, H, W, 3)

        # Pad pred to same length as gt if needed
        if pred_np.shape[0] < total_len:
            pad = np.zeros((total_len - pred_np.shape[0], *pred_np.shape[1:]), dtype=np.uint8)
            pred_np = np.concatenate([pred_np, pad], axis=0)

        # --- Metrics ---
        min_len = min(total_len - hs, pred_emb.shape[1] - hs)
        gt_fut_emb = np.array(gt_emb[0, hs:hs + min_len])
        pred_fut_emb = np.array(pred_emb[0, hs:hs + min_len])
        emb_mse = float(np.mean((gt_fut_emb - pred_fut_emb) ** 2))

        pixel_mse = float(np.mean(
            (gt_np[hs:hs + min_len].astype(float) - pred_np[hs:hs + min_len].astype(float)) ** 2
        )) / (255 ** 2)

        recon_mse = float(np.mean(
            (gt_np.astype(float) - recon_np.astype(float)) ** 2
        )) / (255 ** 2)

        all_emb_mse.append(emb_mse)
        all_pixel_mse.append(pixel_mse)

        print(f"  Example {ex}: emb_MSE={emb_mse:.6f}, pixel_MSE={pixel_mse:.6f}, recon_MSE={recon_mse:.6f}")

        # Collect for global PCA
        all_gt_embs.append(np.array(gt_emb[0]))
        all_pred_embs.append(np.array(pred_emb[0]))

        # --- 1. Side-by-side GIF ---
        video = make_comparison_video(gt_np, recon_np, pred_np[:total_len], hs)
        imageio.mimwrite(output_dir / f"ex{ex:02d}_video.gif", video, fps=6, loop=0)

        # --- 2. Static comparison PNG ---
        show = min(13, total_len)
        rows = []
        for frames, label_colors in [
            (gt_np, lambda t: [0, 180, 0] if t < hs else [180, 180, 180]),
            (recon_np, lambda t: [0, 100, 220]),
            (pred_np, lambda t: [0, 180, 0] if t < hs else [220, 120, 0]),
        ]:
            row = np.concatenate([add_border(frames[t], label_colors(t)) for t in range(show)], axis=1)
            rows.append(row)
        comparison = np.concatenate(rows, axis=0)
        imageio.imwrite(output_dir / f"ex{ex:02d}_comparison.png", comparison)

        # --- 3. Latent PCA ---
        plot_latent_pca(
            np.array(gt_emb[0]),
            np.array(pred_emb[0]),
            hs,
            output_dir / f"ex{ex:02d}_latent.png",
        )

    # --- Global summary ---
    if all_emb_mse:
        print(f"\n{'='*60}")
        print(f"SUMMARY ({len(all_emb_mse)} examples, {config.rollout_steps}-step rollout)")
        print(f"{'='*60}")
        print(f"  Embedding MSE: {np.mean(all_emb_mse):.6f} ± {np.std(all_emb_mse):.6f}")
        print(f"  Pixel MSE:     {np.mean(all_pixel_mse):.6f} ± {np.std(all_pixel_mse):.6f}")

    # --- Global PCA: all examples together ---
    if len(all_gt_embs) >= 2:
        fig, ax = plt.subplots(figsize=(8, 6))
        all_concat = np.concatenate(all_gt_embs + all_pred_embs, axis=0)
        pca = PCA(n_components=2)
        all_2d = pca.fit_transform(all_concat)

        offset = 0
        colors = plt.cm.tab10(np.linspace(0, 1, len(all_gt_embs)))
        for i in range(len(all_gt_embs)):
            n_gt = len(all_gt_embs[i])
            n_pred = len(all_pred_embs[i])
            gt_2d = all_2d[offset:offset + n_gt]
            offset += n_gt

            ax.plot(gt_2d[:, 0], gt_2d[:, 1], '-', color=colors[i], alpha=0.7, linewidth=1.5)
            ax.scatter(gt_2d[0, 0], gt_2d[0, 1], c=[colors[i]], s=50, marker='s', edgecolors='black', zorder=5)

        for i in range(len(all_pred_embs)):
            n_pred = len(all_pred_embs[i])
            pred_2d = all_2d[offset:offset + n_pred]
            offset += n_pred

            ax.plot(pred_2d[hs:, 0], pred_2d[hs:, 1], '--', color=colors[i], alpha=0.5, linewidth=1)

        ax.set_title(f'Global Latent Space (PCA, var={pca.explained_variance_ratio_.sum():.1%})')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.grid(True, alpha=0.3)

        # Custom legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='gray', linewidth=2, label='GT trajectory'),
            Line2D([0], [0], color='gray', linewidth=1, linestyle='--', label='WM predicted'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='gray',
                   markersize=8, label='Start'),
        ]
        ax.legend(handles=legend_elements, fontsize=9)
        plt.tight_layout()
        plt.savefig(output_dir / "global_latent_pca.png", dpi=120, bbox_inches='tight')
        plt.close()
        print(f"\n  Global PCA saved to {output_dir}/global_latent_pca.png")

    print(f"\nAll outputs saved to {output_dir}/")
    print(f"  *_video.gif     — GT (top) / Recon (mid) / Predicted (bottom)")
    print(f"  *_comparison.png — static frame comparison")
    print(f"  *_latent.png     — PCA trajectory + MSE over time")


if __name__ == "__main__":
    tyro.cli(main)
