"""Evaluate the JEPA world model on kinetix tasks using MPC (CEM solver).

Usage:
    uv run src/eval_world_model.py \
        --wm-checkpoint-dir ./logs-wm/<run-name>/99 \
        --level-paths worlds/l/grasp_easy.json

The evaluation loop:
  1. Loads a trained world model checkpoint
  2. Resets kinetix environment to a level
  3. At each step, uses CEM to plan action sequences that minimize
     cost in the world model's latent space
  4. Executes the best action and records metrics
"""

import dataclasses
import functools
import pathlib
import pickle
from typing import Sequence

import einops
import flax.nnx as nnx
import imageio
import jax
import jax.numpy as jnp
import kinetix.environment.env as kenv
import kinetix.environment.env_state as kenv_state
import kinetix.environment.wrappers as wrappers
import kinetix.render.renderer_pixels as renderer_pixels
import numpy as np
import tqdm_loggable.auto as tqdm
import tyro

import train_expert
import world_model as wm


@dataclasses.dataclass(frozen=True)
class CEMConfig:
    """Cross-Entropy Method configuration for MPC planning."""
    num_samples: int = 128       # Number of action candidates per iteration
    num_iterations: int = 5      # CEM iterations
    top_k: int = 16              # Keep top-k candidates
    planning_horizon: int = 16   # Steps to plan ahead
    action_std: float = 0.3      # Initial action sampling std


@dataclasses.dataclass(frozen=True)
class EvalConfig:
    wm_checkpoint_dir: str
    level_paths: Sequence[str] = (
        "worlds/l/grasp_easy.json",
    )
    seed: int = 0
    num_episodes: int = 32       # Number of evaluation episodes
    cem: CEMConfig = CEMConfig()
    output_dir: str = "eval_wm_output"
    render_video: bool = True


def cem_plan(
    rng: jax.Array,
    model: wm.JEPA,
    pixel_history: jax.Array,
    action_history: jax.Array,
    goal_pixels: jax.Array,
    config: CEMConfig,
    action_dim: int,
) -> jax.Array:
    """Plan actions using Cross-Entropy Method.

    Args:
        rng: PRNG key
        model: trained JEPA world model
        pixel_history: (history_size, H, W, C) recent pixel observations
        action_history: (history_size, action_dim) recent actions
        goal_pixels: (H, W, C) goal frame
        config: CEM configuration
        action_dim: action dimension

    Returns:
        best_action: (action_dim,) the first action from the best plan
    """
    h = config.planning_horizon

    # Initialize mean and std for action sampling
    mean = jnp.zeros((h, action_dim))
    std = jnp.ones((h, action_dim)) * config.action_std

    # Add batch dimension for model input
    ctx_pixels = pixel_history[None]       # (1, T, H, W, C)
    ctx_actions = action_history[None]     # (1, T, action_dim)
    goal = goal_pixels[None, None]         # (1, 1, H, W, C)

    def cem_iteration(carry, _):
        rng, mean, std = carry
        rng, sample_rng = jax.random.split(rng)

        # Sample action candidates: (num_samples, horizon, action_dim)
        noise = jax.random.normal(sample_rng, (config.num_samples, h, action_dim))
        candidates = mean[None] + noise * std[None]
        candidates = jnp.clip(candidates, -1.0, 1.0)

        # Evaluate candidates through world model
        # Expand context for each candidate
        ctx_px = jnp.broadcast_to(ctx_pixels, (config.num_samples, *ctx_pixels.shape[1:]))
        ctx_act = jnp.broadcast_to(ctx_actions, (config.num_samples, *ctx_actions.shape[1:]))

        # Rollout world model for each candidate
        all_emb = jax.vmap(
            lambda px, act, fut: model.rollout(px[None], act[None], fut[None])[0]
        )(ctx_px, ctx_act, candidates)  # (num_samples, T+N+1, D)

        # Encode goal
        goal_expanded = jnp.broadcast_to(goal, (config.num_samples, *goal.shape[1:]))
        goal_emb, _ = jax.vmap(
            lambda g: model.encode(g[None], jnp.zeros((1, 1, action_dim)))
        )(goal_expanded)  # (num_samples, 1, D)
        goal_emb = goal_emb[:, 0, 0]  # (num_samples, D)

        # Cost: MSE between final predicted embedding and goal embedding
        final_emb = all_emb[:, -1]  # (num_samples, D)
        costs = jnp.mean((final_emb - goal_emb) ** 2, axis=-1)  # (num_samples,)

        # Select top-k
        top_k_idxs = jnp.argsort(costs)[:config.top_k]
        elite = candidates[top_k_idxs]  # (top_k, horizon, action_dim)

        # Update distribution
        new_mean = jnp.mean(elite, axis=0)
        new_std = jnp.std(elite, axis=0) + 1e-6

        return (rng, new_mean, new_std), costs.min()

    (rng, final_mean, final_std), min_costs = jax.lax.scan(
        cem_iteration, (rng, mean, std), None, length=config.num_iterations
    )

    # Return first action of best plan
    return final_mean[0]


def evaluate_level(
    rng: jax.Array,
    model: wm.JEPA,
    env,
    level: kenv_state.EnvState,
    env_params: kenv_state.EnvParams,
    static_env_params: kenv_state.StaticEnvParams,
    config: EvalConfig,
    render_pixels_fn,
):
    """Evaluate world model on a single level using MPC."""
    action_dim = env.action_space(env_params).shape[0]
    model_config = model.config
    hs = model_config.history_size

    results = {
        "episode_returns": [],
        "episode_solved": [],
        "episode_lengths": [],
    }
    all_frames = []

    for ep in range(config.num_episodes):
        rng, reset_rng = jax.random.split(rng)
        obs, env_state = env.reset_to_level(reset_rng, level, env_params)

        # Render initial frame
        def get_pixels(state):
            s = state
            while not isinstance(s, kenv_state.EnvState):
                s = s.env_state
            return render_pixels_fn(s).round().astype(jnp.uint8).transpose(1, 0, 2)[::-1]

        # Initialize history buffers
        init_pixels = get_pixels(env_state)  # (H, W, 3)
        pixel_history = jnp.broadcast_to(
            init_pixels[None].astype(jnp.float32) / 255.0,
            (hs, *init_pixels.shape)
        ).astype(jnp.float32)  # already /255
        # Redo: keep as float [0,1]
        pixel_history = jnp.broadcast_to(
            (init_pixels.astype(jnp.float32) / 255.0)[None],
            (hs, *init_pixels.shape)
        )
        action_history = jnp.zeros((hs, action_dim))

        episode_return = 0.0
        episode_length = 0
        solved = False
        frames = [np.array(init_pixels)]

        for step in range(env_params.max_timesteps):
            rng, plan_rng, step_rng = jax.random.split(rng, 3)

            # For now, use random goal (last frame as goal — self-supervised)
            # In practice, you'd set a specific goal state
            goal_pixels = pixel_history[-1]  # Use current frame as placeholder

            # Plan action via CEM
            action = cem_plan(
                plan_rng,
                model,
                pixel_history,
                action_history,
                goal_pixels,
                config.cem,
                action_dim,
            )
            action = jnp.clip(action, -0.99999, 0.99999)

            # Step environment
            next_obs, next_env_state, reward, done, info = env.step(
                step_rng, env_state, action, env_params
            )

            episode_return += float(reward)
            episode_length += 1

            # Render and update history
            next_pixels = get_pixels(next_env_state)
            frames.append(np.array(next_pixels))

            pixel_history = jnp.concatenate([
                pixel_history[1:],
                (next_pixels.astype(jnp.float32) / 255.0)[None],
            ], axis=0)
            action_history = jnp.concatenate([
                action_history[1:],
                action[None],
            ], axis=0)

            env_state = next_env_state
            obs = next_obs

            if done:
                solved = bool(info.get("returned_episode_solved", False))
                break

        results["episode_returns"].append(episode_return)
        results["episode_solved"].append(solved)
        results["episode_lengths"].append(episode_length)

        if config.render_video and ep < 4:
            all_frames.append(frames)

        print(f"  Episode {ep}: return={episode_return:.2f}, solved={solved}, length={episode_length}")

    return results, all_frames


def main(config: EvalConfig):
    print(f"Evaluating world model from {config.wm_checkpoint_dir}")

    # Load world model
    ckpt_dir = pathlib.Path(config.wm_checkpoint_dir)
    with (ckpt_dir / "config.pkl").open("rb") as f:
        model_config = pickle.load(f)
    with (ckpt_dir / "world_model.pkl").open("rb") as f:
        state_dict = pickle.load(f)

    rng = jax.random.key(config.seed)
    rng, init_rng = jax.random.split(rng)
    model = wm.JEPA(model_config, rngs=nnx.Rngs(init_rng))

    # Load weights
    graphdef, state = nnx.split(model)
    state.replace_by_pure_dict(state_dict)
    model = nnx.merge(graphdef, state)
    print("Model loaded successfully")

    # Setup environment
    static_env_params = kenv_state.StaticEnvParams(
        **train_expert.LARGE_ENV_PARAMS, frame_skip=train_expert.FRAME_SKIP
    )
    env_params = kenv_state.EnvParams()
    static_env_params = static_env_params.replace(
        screen_dim=(model_config.img_size, model_config.img_size)
    )

    env = kenv.make_kinetix_env_from_name(
        "Kinetix-Symbolic-Continuous-v1", static_env_params=static_env_params
    )
    env = wrappers.AutoReplayWrapper(train_expert.NoisyActionWrapper(env))

    render_pixels_fn = renderer_pixels.make_render_pixels(env_params, static_env_params)

    # Evaluate each level
    output_dir = pathlib.Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for level_path in config.level_paths:
        level_name = level_path.replace("/", "_").replace(".json", "")
        print(f"\nEvaluating {level_name}...")

        levels = train_expert.load_levels([level_path], static_env_params, env_params)
        level = jax.tree.map(lambda x: x[0], levels)

        rng, eval_rng = jax.random.split(rng)
        results, frames = evaluate_level(
            eval_rng, model, env, level, env_params, static_env_params,
            config, render_pixels_fn,
        )

        all_results[level_name] = {
            "mean_return": np.mean(results["episode_returns"]),
            "mean_solved": np.mean(results["episode_solved"]),
            "mean_length": np.mean(results["episode_lengths"]),
        }

        print(f"  {level_name}: return={all_results[level_name]['mean_return']:.2f}, "
              f"solved={all_results[level_name]['mean_solved']:.2%}")

        # Save videos
        if config.render_video and frames:
            video_dir = output_dir / "videos"
            video_dir.mkdir(parents=True, exist_ok=True)
            for ep_idx, ep_frames in enumerate(frames):
                imageio.mimwrite(
                    video_dir / f"{level_name}_ep{ep_idx}.gif",
                    ep_frames,
                    fps=15,
                )

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    for name, res in all_results.items():
        print(f"  {name}: return={res['mean_return']:.2f}, solved={res['mean_solved']:.2%}")

    # Save results
    with (output_dir / "results.pkl").open("wb") as f:
        pickle.dump(all_results, f)
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    tyro.cli(main)
