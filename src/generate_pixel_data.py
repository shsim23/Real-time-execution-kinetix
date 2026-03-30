"""Generate pixel + action trajectory data from expert policies for world model training.

Usage:
    uv run src/generate_pixel_data.py --config.run-path ./logs-expert

Processes one level at a time to avoid OOM. Each level's data is saved
independently as a compressed npz file.
"""

import dataclasses
import functools
import gc
import pathlib
import pickle
from typing import Sequence

import einops
from flax import struct
import flax.nnx as nnx
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


@dataclasses.dataclass
class Config:
    run_path: str
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
    num_envs: int = 32
    batch_size: int = 64
    num_steps: int = 100_000
    pixel_size: int = 64
    output_dir: str = "data-pixel"


@struct.dataclass
class PixelData:
    pixels: jax.Array    # (steps, envs, H, W, 3) uint8
    obs: jax.Array       # (steps, envs, obs_dim) float32
    action: jax.Array    # (steps, envs, action_dim) float32
    done: jax.Array      # (steps, envs) bool
    solved: jax.Array    # (steps, envs) float32
    return_: jax.Array   # (steps, envs) float32
    length: jax.Array    # (steps, envs) float32


@struct.dataclass
class StepCarry:
    rng: jax.Array
    obs: jax.Array
    env_state: kenv_state.EnvState
    policy_idxs: jax.Array


def generate_level(
    config: Config,
    level_path: str,
    env,
    env_wrapped,
    env_params: kenv_state.EnvParams,
    static_env_params: kenv_state.StaticEnvParams,
    render_batch,
    rng: jax.Array,
):
    """Generate pixel data for a single level."""
    level_name = level_path.replace("/", "_").replace(".json", "")
    print(f"\n{'='*60}")
    print(f"Processing: {level_name}")
    print(f"{'='*60}")

    # Load level
    level, _, _ = __import__("kinetix.util.saving", fromlist=["load_from_json_file"]).load_from_json_file(level_path)

    # Load expert policies for this level only
    state_dicts_list = []
    seed_dirs = sorted(pathlib.Path(config.run_path).glob("seed_*"))
    if len(seed_dirs) == 0:
        seed_dirs = [pathlib.Path(config.run_path)]

    for seed_dir in seed_dirs:
        log_dirs = sorted(
            [p for p in seed_dir.iterdir() if p.is_dir() and p.name.isdigit()],
            key=lambda p: int(p.name),
        )
        chosen_log_dir = log_dirs[-1]
        with open(chosen_log_dir / "policies" / f"{level_name}.pkl", "rb") as f:
            state_dicts_list.append(pickle.load(f))
        print(f"\tLoaded from {chosen_log_dir}")

    # Stack into single dict with leading seed dimension
    state_dict = jax.device_put(jax.tree.map(lambda *x: jnp.array(x), *state_dicts_list))
    good_policy_mask = jnp.ones(len(state_dicts_list), dtype=bool)
    del state_dicts_list

    num_steps_per_env = (
        (config.num_steps // config.num_envs + config.batch_size - 1) // config.batch_size
    ) * config.batch_size

    def new_policy_idxs(rng):
        rng, key = jax.random.split(rng)
        n_policies = good_policy_mask.sum()
        randint = jax.random.randint(key, (config.num_envs,), 0, n_policies)
        return jnp.nonzero(good_policy_mask, size=good_policy_mask.shape[0])[0][randint]

    @jax.jit
    def init(rng):
        rng, key = jax.random.split(rng)
        obs, env_state = env_wrapped.reset_to_level(key, level, env_params)
        rng, key = jax.random.split(rng)
        policy_idxs = new_policy_idxs(key)
        return StepCarry(rng, obs, env_state, policy_idxs)

    @functools.partial(jax.jit, static_argnums=(2,), donate_argnums=(0,))
    def step_n(carry: StepCarry, state_dict: dict, n: int):
        def step(carry: StepCarry, _):
            action_dim = env_wrapped.action_space(env_params).shape[0]
            obs_dim = carry.obs.shape[1]

            @jax.vmap
            def get_action(key, obs, policy_idx):
                agent = train_expert.Agent(obs_dim, action_dim, 1, rngs=nnx.Rngs(0))
                graphdef, state = nnx.split(agent)
                state.replace_by_pure_dict(jax.tree.map(lambda x: x[policy_idx], state_dict))
                agent = nnx.merge(graphdef, state)
                mean, std = agent.action(obs)
                action_dist = train_expert.make_squashed_normal_diag(
                    mean, std, static_env_params.num_motor_bindings
                )
                return action_dist.sample(seed=key)

            rng, key = jax.random.split(carry.rng)
            action = get_action(jax.random.split(key, config.num_envs), carry.obs, carry.policy_idxs)
            rng, key = jax.random.split(rng)
            next_obs, next_env_state, reward, done, info = env_wrapped.step(
                key, carry.env_state, action, env_params
            )

            rng, key = jax.random.split(rng)
            next_policy_idxs = jnp.where(done, new_policy_idxs(key), carry.policy_idxs)

            # Render pixels from current state
            pixels = render_batch(carry.env_state)  # (num_envs, H, W, 3)

            return StepCarry(rng, next_obs, next_env_state, next_policy_idxs), PixelData(
                pixels=pixels,
                obs=train_expert.ObsHistoryWrapper.get_original_obs(carry.env_state),
                action=action,
                done=done,
                solved=info["returned_episode_solved"],
                return_=info["returned_episode_returns"],
                length=info["returned_episode_lengths"],
            )

        return jax.lax.scan(step, carry, None, length=n)

    # Run data collection
    carry = init(rng)
    pbar = tqdm.tqdm(
        total=num_steps_per_env * config.num_envs,
        desc=level_name,
        dynamic_ncols=True,
    )
    data_chunks = []
    for _ in range(0, num_steps_per_env, config.batch_size):
        carry, result = step_n(carry, state_dict, config.batch_size)
        data_chunks.append(jax.device_get(result))
        pbar.update(config.batch_size * config.num_envs)
    pbar.close()

    # Merge chunks: each chunk is (batch_size, num_envs, ...)
    # Stack into (num_batches, batch_size, num_envs, ...) then reshape
    with jax.default_device(jax.devices("cpu")[0]):
        data: PixelData = jax.tree.map(
            lambda *x: np.concatenate(x, axis=0),
            *data_chunks,
        )
    del data_chunks

    # Print stats
    num_episodes = data.done.sum()
    solved_rate = (data.solved * data.done).sum() / max(num_episodes, 1)
    avg_return = (data.return_ * data.done).sum() / max(num_episodes, 1)
    avg_length = (data.length * data.done).sum() / max(num_episodes, 1)
    print(f"{level_name}:")
    print(f"\tnum_episodes: {num_episodes:.0f}")
    print(f"\tsolved: {solved_rate:.3f}")
    print(f"\treturn: {avg_return:.3f}")
    print(f"\tlength: {avg_length:.1f}")

    # Save
    output_dir = pathlib.Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    level_data = {
        "pixels": np.array(data.pixels),
        "obs": np.array(data.obs),
        "action": np.array(data.action),
        "done": np.array(data.done),
        "solved": np.array(data.solved),
    }
    save_path = output_dir / f"{level_name}.npz"
    np.savez_compressed(save_path, **level_data)
    print(f"\tSaved to {save_path}")
    print(f"\tPixels shape: {level_data['pixels'].shape}")

    # Free memory
    del data, level_data, state_dict, carry
    gc.collect()


def main(config: Config):
    print(f"Generating pixel data for {len(config.level_paths)} levels")
    print(f"  num_steps={config.num_steps:_}, num_envs={config.num_envs}, pixel_size={config.pixel_size}")

    static_env_params = kenv_state.StaticEnvParams(
        **train_expert.LARGE_ENV_PARAMS, frame_skip=train_expert.FRAME_SKIP
    )
    env_params = kenv_state.EnvParams()

    # screen_dim / downscale = actual pixel resolution
    downscale = static_env_params.downscale  # default 4
    screen_dim = (config.pixel_size * downscale, config.pixel_size * downscale)
    static_env_params = static_env_params.replace(screen_dim=screen_dim)

    env = kenv.make_kinetix_env_from_name(
        "Kinetix-Symbolic-Continuous-v1", static_env_params=static_env_params
    )

    # Pixel renderer
    render_pixels_fn = renderer_pixels.make_render_pixels(env_params, static_env_params)

    def render_single_state(env_state):
        state = env_state
        while not isinstance(state, kenv_state.EnvState):
            state = state.env_state
        pixels = render_pixels_fn(state)
        pixels = pixels.round().astype(jnp.uint8).transpose(1, 0, 2)[::-1]
        return pixels

    render_batch = jax.vmap(render_single_state)

    env_wrapped = train_expert.BatchEnvWrapper(
        wrappers.LogWrapper(
            wrappers.AutoReplayWrapper(
                train_expert.ActionHistoryWrapper(
                    train_expert.ObsHistoryWrapper(train_expert.NoisyActionWrapper(env), 4)
                )
            )
        ),
        config.num_envs,
    )

    # Process each level sequentially to avoid OOM
    rng = jax.random.key(config.seed)
    for level_path in config.level_paths:
        output_dir = pathlib.Path(config.output_dir)
        level_name = level_path.replace("/", "_").replace(".json", "")
        if (output_dir / f"{level_name}.npz").exists():
            print(f"Skipping {level_name} (already exists)")
            continue

        rng, level_rng = jax.random.split(rng)
        generate_level(
            config, level_path, env, env_wrapped,
            env_params, static_env_params, render_batch, level_rng,
        )

    print("\nAll levels complete!")


if __name__ == "__main__":
    tyro.cli(main)
