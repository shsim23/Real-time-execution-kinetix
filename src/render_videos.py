"""Render videos of specific tasks to visualize reactiveness."""

import dataclasses
import functools
import pathlib
import pickle
import imageio
from typing import Sequence

import flax.nnx as nnx
import jax
from jax.experimental import shard_map
import jax.numpy as jnp
import kinetix.environment.env as kenv
import kinetix.environment.env_state as kenv_state

import eval_flow as _eval
import model as _model
import train_expert

@dataclasses.dataclass(frozen=True)
class RenderConfig:
    step: int = -1
    num_evals: int = 1  # Only 1 episode per level
    num_flow_steps: int = 5
    inference_delay: int = 2
    execute_horizon: int = 3
    model: _model.ModelConfig = _model.ModelConfig()

def main(
    run_path: str = "./logs-bc/bc",
    config: RenderConfig = RenderConfig(),
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
    ),
    seed: int = 0,
    output_dir: str = "videos",
):
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    static_env_params = kenv_state.StaticEnvParams(**train_expert.LARGE_ENV_PARAMS, frame_skip=train_expert.FRAME_SKIP)
    env_params = kenv_state.EnvParams()
    levels = train_expert.load_levels(level_paths, static_env_params, env_params)
    static_env_params = static_env_params.replace(screen_dim=train_expert.SCREEN_DIM)

    env = kenv.make_kinetix_env_from_name("Kinetix-Symbolic-Continuous-v1", static_env_params=static_env_params)

    # Load policies
    state_dicts = []
    for level_path in level_paths:
        level_name = level_path.replace("/", "_").replace(".json", "")
        log_dirs = list(filter(lambda p: p.is_dir() and p.name.isdigit(), pathlib.Path(run_path).iterdir()))
        log_dirs = sorted(log_dirs, key=lambda p: int(p.name))
        with (log_dirs[config.step] / "policies" / f"{level_name}.pkl").open("rb") as f:
            state_dicts.append(pickle.load(f))
    state_dicts = jax.device_put(jax.tree.map(lambda *x: jnp.array(x), *state_dicts))

    obs_dim = jax.eval_shape(env.reset_to_level, jax.random.key(0), jax.tree.map(lambda x: x[0], levels), env_params)[0].shape[-1]
    action_dim = env.action_space(env_params).shape[0]

    mesh = jax.make_mesh((jax.local_device_count(),), ("x",))
    pspec = jax.sharding.PartitionSpec("x")
    sharding = jax.sharding.NamedSharding(mesh, pspec)

    weak_state_dicts = jax.tree.map(jnp.zeros_like, state_dicts)

    @functools.partial(jax.jit, static_argnums=(0,), in_shardings=sharding, out_shardings=sharding)
    @functools.partial(shard_map.shard_map, mesh=mesh, in_specs=(None, pspec, pspec, pspec, pspec), out_specs=pspec)
    @functools.partial(jax.vmap, in_axes=(None, 0, 0, 0, 0))
    def _eval_fn(eval_config, rng, level, state_dict, weak_state_dict):
        policy = _model.FlowPolicy(obs_dim=obs_dim, action_dim=action_dim, config=eval_config.model, rngs=nnx.Rngs(rng))
        graphdef, state = nnx.split(policy)
        state.replace_by_pure_dict(state_dict)
        policy = nnx.merge(graphdef, state)
        eval_info, video = _eval.eval(eval_config, env, rng, level, policy, env_params, static_env_params, None)
        return video

    rngs = jax.random.split(jax.random.key(seed), len(level_paths))

    method = _eval.RealtimeMethodConfig(prefix_attention_schedule="exp", max_guidance_weight=5.0, exp_decay_alpha=5.0)
    
    eval_config = _eval.EvalConfig(
        step=config.step,
        num_evals=config.num_evals,
        num_flow_steps=config.num_flow_steps,
        inference_delay=config.inference_delay,
        execute_horizon=config.execute_horizon,
        method=method,
        model=config.model,
    )

    print("Rendering videos...")
    video_outputs = jax.device_get(_eval_fn(eval_config, rngs, levels, state_dicts, weak_state_dicts))
    
    for i, level_path in enumerate(level_paths):
        level_name = level_path.replace('worlds/l/', '').replace('.json', '')
        out_path = pathlib.Path(output_dir) / f"{level_name}.gif"
        # Output as GIF for foolproof markdown/editor playback
        imageio.mimwrite(out_path, video_outputs[i], fps=15, format='GIF', loop=0)
        print(f"Saved {out_path}")

if __name__ == "__main__":
    import tyro
    tyro.cli(main)
