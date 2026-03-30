"""Train the JEPA world model on kinetix pixel+action trajectory data.

Usage:
    uv run src/train_world_model.py --config.data-dir ./data-pixel --config.level-paths worlds/l/grasp_easy.json

The training loop:
  1. Loads pixel+action trajectories from generate_pixel_data.py output
  2. Samples random context windows (history_size + num_preds frames)
  3. Trains JEPA with MSE prediction loss + SIGReg regularizer
  4. Logs metrics to wandb and saves checkpoints
"""

import concurrent.futures
import dataclasses
import functools
import pathlib
import pickle
from typing import Sequence

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import tyro
import wandb

import world_model as wm


WANDB_PROJECT = "rtc-kinetix-wm"
LOG_DIR = pathlib.Path("logs-wm")


@dataclasses.dataclass(frozen=True)
class Config:
    data_dir: str = "data-pixel"
    level_paths: Sequence[str] = (
        "worlds/l/grasp_easy.json",
        "worlds/l/catapult.json",
        "worlds/l/cartpole_thrust.json",
        "worlds/l/hard_lunar_lander.json",
        "worlds/l/mjc_half_cheetah.json",
        "worlds/l/mjc_swimmer.json",
        "worlds/l/mjc_walker.json",
        "worlds/l/h17_unicycle.json",
        "worlds/l/chain_lander.json",
        "worlds/l/catcher_v3.json",
        "worlds/l/trampoline.json",
        "worlds/l/car_launch.json",
    )
    seed: int = 0
    batch_size: int = 64
    num_epochs: int = 100
    learning_rate: float = 5e-5
    weight_decay: float = 1e-3
    grad_norm_clip: float = 1.0
    lr_warmup_steps: int = 500

    # World model config
    model: wm.WorldModelConfig = wm.WorldModelConfig()

    # Evaluation
    eval_every: int = 5  # epochs
    save_every: int = 10  # epochs

    # Data loading
    num_envs_to_use: int | None = None  # None = use all
    max_steps: int | None = None  # None = use all


@dataclasses.dataclass
class TrajectoryDataset:
    """Dataset of pixel+action trajectories for a single level."""
    pixels: np.ndarray    # (steps, envs, H, W, 3) uint8
    obs: np.ndarray       # (steps, envs, obs_dim) float32
    actions: np.ndarray   # (steps, envs, action_dim) float32
    done: np.ndarray      # (steps, envs) bool

    @staticmethod
    def load(path: pathlib.Path, num_envs: int | None = None, max_steps: int | None = None):
        data = np.load(path)
        pixels = data["pixels"]
        obs = data["obs"]
        actions = data["action"]
        done = data["done"]

        if num_envs is not None:
            pixels = pixels[:, :num_envs]
            obs = obs[:, :num_envs]
            actions = actions[:, :num_envs]
            done = done[:, :num_envs]

        if max_steps is not None:
            pixels = pixels[:max_steps]
            obs = obs[:max_steps]
            actions = actions[:max_steps]
            done = done[:max_steps]

        return TrajectoryDataset(pixels=pixels, obs=obs, actions=actions, done=done)

    @property
    def num_steps(self) -> int:
        return self.pixels.shape[0]

    @property
    def num_envs(self) -> int:
        return self.pixels.shape[1]


def sample_context_windows(
    rng: jax.Array,
    pixels: jax.Array,
    actions: jax.Array,
    done: jax.Array,
    window_size: int,
    batch_size: int,
) -> tuple[jax.Array, jax.Array]:
    """Sample random context windows from trajectory data.

    Returns:
        pixel_windows: (batch_size, window_size, H, W, C) float32 [0, 1]
        action_windows: (batch_size, window_size, action_dim) float32
    """
    num_steps, num_envs = pixels.shape[:2]
    max_start = num_steps - window_size

    rng1, rng2 = jax.random.split(rng)
    # Sample random start indices and env indices
    start_idxs = jax.random.randint(rng1, (batch_size,), 0, max_start)
    env_idxs = jax.random.randint(rng2, (batch_size,), 0, num_envs)

    # Gather windows
    offsets = jnp.arange(window_size)

    def gather_window(start_idx, env_idx):
        step_idxs = start_idx + offsets
        pix = pixels[step_idxs, env_idx]    # (window_size, H, W, C)
        act = actions[step_idxs, env_idx]    # (window_size, action_dim)
        d = done[step_idxs, env_idx]         # (window_size,)
        return pix, act, d

    pixel_windows, action_windows, done_windows = jax.vmap(gather_window)(start_idxs, env_idxs)

    # Normalize pixels to [0, 1]
    pixel_windows = pixel_windows.astype(jnp.float32) / 255.0

    # Zero out actions after done (episode boundary)
    has_done = jnp.cumsum(done_windows, axis=-1) > 0
    # Shift: done at step t means step t+1 onwards should be zeroed
    has_done_shifted = jnp.concatenate(
        [jnp.zeros((batch_size, 1), dtype=bool), has_done[:, :-1]], axis=1
    )
    action_windows = jnp.where(has_done_shifted[:, :, None], 0.0, action_windows)

    return pixel_windows, action_windows


def main(config: Config):
    print(f"Training world model with config:\n{config}")

    # Load data for all levels
    data_dir = pathlib.Path(config.data_dir)
    datasets = {}

    def load_level(level_path: str):
        level_name = level_path.replace("/", "_").replace(".json", "")
        path = data_dir / f"{level_name}.npz"
        if not path.exists():
            print(f"WARNING: {path} not found, skipping level {level_name}")
            return level_name, None
        ds = TrajectoryDataset.load(path, config.num_envs_to_use, config.max_steps)
        print(f"Loaded {level_name}: pixels={ds.pixels.shape}, actions={ds.actions.shape}")
        return level_name, ds

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(load_level, config.level_paths))

    for name, ds in results:
        if ds is not None:
            datasets[name] = ds

    if not datasets:
        raise ValueError(f"No datasets found in {data_dir}. Run generate_pixel_data.py first.")

    # Infer action dim from data
    first_ds = next(iter(datasets.values()))
    action_dim = first_ds.actions.shape[-1]
    print(f"Action dim: {action_dim}, Pixel shape: {first_ds.pixels.shape[2:]}")

    # Update config with correct action_dim
    model_config = dataclasses.replace(config.model, action_dim=action_dim)
    window_size = model_config.history_size + model_config.num_preds

    # Initialize model
    rng = jax.random.key(config.seed)
    rng, init_rng = jax.random.split(rng)
    model = wm.JEPA(model_config, rngs=nnx.Rngs(init_rng))

    # Count parameters
    total_params = sum(x.size for x in jax.tree.leaves(nnx.state(model, nnx.Param)))
    print(f"Total parameters: {total_params:,}")

    # Optimizer
    schedule = optax.warmup_constant_schedule(0, config.learning_rate, config.lr_warmup_steps)
    optimizer = nnx.Optimizer(
        model,
        optax.chain(
            optax.clip_by_global_norm(config.grad_norm_clip),
            optax.adamw(schedule, weight_decay=config.weight_decay),
        ),
    )

    # Training step
    @nnx.jit
    def train_step(
        model: wm.JEPA,
        optimizer: nnx.Optimizer,
        rng: jax.Array,
        pixel_batch: jax.Array,
        action_batch: jax.Array,
    ):
        def loss_fn(model):
            return model.loss(rng, pixel_batch, action_batch)

        (loss, info), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
        grad_norm = optax.global_norm(grads)
        optimizer.update(grads)
        info["grad_norm"] = grad_norm
        return info

    # Wandb init
    wandb.init(project=WANDB_PROJECT, config=dataclasses.asdict(config))
    log_dir = LOG_DIR / wandb.run.name
    log_dir.mkdir(parents=True, exist_ok=True)

    # Merge all level data for mixed training
    all_pixels = []
    all_actions = []
    all_done = []
    level_names = []
    for name, ds in datasets.items():
        # Flatten envs into steps: (steps, envs, ...) -> (steps*envs, ...)
        # Actually keep (steps, envs, ...) for window sampling
        all_pixels.append(ds.pixels)
        all_actions.append(ds.actions)
        all_done.append(ds.done)
        level_names.append(name)

    print(f"\nTraining on {len(datasets)} levels, window_size={window_size}")
    print(f"Batch size: {config.batch_size}, Epochs: {config.num_epochs}")

    global_step = 0
    for epoch in range(config.num_epochs):
        epoch_losses = []

        # Train on each level's data
        for level_idx, (name, ds) in enumerate(datasets.items()):
            num_batches = max(1, (ds.num_steps * ds.num_envs) // config.batch_size // window_size)

            # Put data on device (keep as numpy until sampling)
            pixels_jax = jnp.array(ds.pixels)
            actions_jax = jnp.array(ds.actions)
            done_jax = jnp.array(ds.done)

            for batch_idx in range(num_batches):
                rng, sample_rng, step_rng = jax.random.split(rng, 3)

                pixel_batch, action_batch = sample_context_windows(
                    sample_rng,
                    pixels_jax,
                    actions_jax,
                    done_jax,
                    window_size,
                    config.batch_size,
                )

                info = train_step(model, optimizer, step_rng, pixel_batch, action_batch)
                epoch_losses.append(float(info["total_loss"]))

                if global_step % 50 == 0:
                    wandb.log(
                        {
                            f"{name}/pred_loss": float(info["pred_loss"]),
                            f"{name}/sigreg_loss": float(info["sigreg_loss"]),
                            f"{name}/total_loss": float(info["total_loss"]),
                            f"{name}/grad_norm": float(info["grad_norm"]),
                            "global_step": global_step,
                        },
                        step=global_step,
                    )

                global_step += 1

        avg_loss = np.mean(epoch_losses)
        print(f"Epoch {epoch}/{config.num_epochs} — avg loss: {avg_loss:.6f}")
        wandb.log({"epoch": epoch, "avg_loss": avg_loss}, step=global_step)

        # Save checkpoint
        if (epoch + 1) % config.save_every == 0 or epoch == config.num_epochs - 1:
            ckpt_dir = log_dir / str(epoch)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            state_dict = nnx.state(model).to_pure_dict()
            with (ckpt_dir / "world_model.pkl").open("wb") as f:
                pickle.dump(state_dict, f)
            # Also save config
            with (ckpt_dir / "config.pkl").open("wb") as f:
                pickle.dump(model_config, f)
            print(f"  Saved checkpoint to {ckpt_dir}")

    wandb.finish()
    print("Training complete!")


if __name__ == "__main__":
    tyro.cli(main)
