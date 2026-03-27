"""Custom evaluation sweep for inpainting control experiments.

Fixed settings: inference_delay=2, execute_horizon=3

Sweeps:
  A) max_guidance_weight [0, 3, 5] with exp_decay_alpha=1.0 (+ naive baseline)
  B) exp_decay_alpha [1.0, 3.0, 5.0] with max_guidance_weight=5.0 (+ hard_masking)

Results are saved to separate subdirectories per config to prevent overwrites.
"""

import collections
import dataclasses
import functools
import math
import pathlib
import pickle
from typing import Sequence

import flax.nnx as nnx
import jax
from jax.experimental import shard_map
import jax.numpy as jnp
import kinetix.environment.env as kenv
import kinetix.environment.env_state as kenv_state
import pandas as pd
import tyro

import eval_flow as _eval
import model as _model
import train_expert


@dataclasses.dataclass(frozen=True)
class SweepConfig:
    step: int = -1
    num_evals: int = 2048
    num_flow_steps: int = 5
    inference_delay: int = 2
    execute_horizon: int = 3
    eval_batch_size: int = 512  # Run in chunks to prevent OOM
    model: _model.ModelConfig = _model.ModelConfig()


def main(
    run_path: str,
    config: SweepConfig = SweepConfig(),
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
    output_dir: str = "eval_output_gw_sweep",
):
    static_env_params = kenv_state.StaticEnvParams(**train_expert.LARGE_ENV_PARAMS, frame_skip=train_expert.FRAME_SKIP)
    env_params = kenv_state.EnvParams()
    levels = train_expert.load_levels(level_paths, static_env_params, env_params)
    static_env_params = static_env_params.replace(screen_dim=train_expert.SCREEN_DIM)

    env = kenv.make_kinetix_env_from_name("Kinetix-Symbolic-Continuous-v1", static_env_params=static_env_params)

    # Load policies from best checkpoints
    state_dicts = []
    for level_path in level_paths:
        level_name = level_path.replace("/", "_").replace(".json", "")
        log_dirs = list(filter(lambda p: p.is_dir() and p.name.isdigit(), pathlib.Path(run_path).iterdir()))
        log_dirs = sorted(log_dirs, key=lambda p: int(p.name))
        with (log_dirs[config.step] / "policies" / f"{level_name}.pkl").open("rb") as f:
            state_dicts.append(pickle.load(f))
    state_dicts = jax.device_put(jax.tree.map(lambda *x: jnp.array(x), *state_dicts))

    obs_dim = jax.eval_shape(env.reset_to_level, jax.random.key(0), jax.tree.map(lambda x: x[0], levels), env_params)[
        0
    ].shape[-1]
    action_dim = env.action_space(env_params).shape[0]

    mesh = jax.make_mesh((jax.local_device_count(),), ("x",))
    pspec = jax.sharding.PartitionSpec("x")
    sharding = jax.sharding.NamedSharding(mesh, pspec)

    # Dummy weak state dicts (not used for our configs)
    weak_state_dicts = jax.tree.map(jnp.zeros_like, state_dicts)

    @functools.partial(jax.jit, static_argnums=(0,), in_shardings=sharding, out_shardings=sharding)
    @functools.partial(shard_map.shard_map, mesh=mesh, in_specs=(None, pspec, pspec, pspec, pspec), out_specs=pspec)
    @functools.partial(jax.vmap, in_axes=(None, 0, 0, 0, 0))
    def _eval_fn(eval_config: _eval.EvalConfig, rng: jax.Array, level: kenv_state.EnvState, state_dict, weak_state_dict):
        policy = _model.FlowPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            config=eval_config.model,
            rngs=nnx.Rngs(rng),
        )
        graphdef, state = nnx.split(policy)
        state.replace_by_pure_dict(state_dict)
        policy = nnx.merge(graphdef, state)
        eval_info, _ = _eval.eval(eval_config, env, rng, level, policy, env_params, static_env_params, None)
        return eval_info

    rngs = jax.random.split(jax.random.key(seed), len(level_paths))

    inference_delay = config.inference_delay
    execute_horizon = config.execute_horizon
    assert execute_horizon >= inference_delay, f"{execute_horizon=} must be >= {inference_delay=}"

    print(f"\nFixed settings: inference_delay={inference_delay}, execute_horizon={execute_horizon}")
    print(f"num_evals={config.num_evals}, num_flow_steps={config.num_flow_steps}")
    print(f"levels: {len(level_paths)}, GPUs: {jax.local_device_count()}\n")

    # ============================================================
    # Define the 7 experiment configs
    # ============================================================
    experiments = []

    # Sweep A: max_guidance_weight [0, 3, 5] with alpha=1.0
    experiments.append({
        "name": "naive",
        "method": _eval.NaiveMethodConfig(),
    })
    for gw in [0, 3, 5]:
        experiments.append({
            "name": f"realtime_gw{gw}_alpha1.0",
            "method": _eval.RealtimeMethodConfig(
                prefix_attention_schedule="exp",
                max_guidance_weight=float(gw),
                exp_decay_alpha=1.0,
            ),
        })

    # Sweep B: exp_decay_alpha [3.0, 5.0] with gw=5.0
    # (alpha=1.0 with gw=5.0 already covered as "realtime_gw5_alpha1.0")
    for alpha in [3.0, 5.0]:
        experiments.append({
            "name": f"realtime_gw5_alpha{alpha}",
            "method": _eval.RealtimeMethodConfig(
                prefix_attention_schedule="exp",
                max_guidance_weight=5.0,
                exp_decay_alpha=alpha,
            ),
        })

    # Hard masking (frozen only, zeros elsewhere)
    experiments.append({
        "name": "hard_masking_gw5",
        "method": _eval.RealtimeMethodConfig(
            prefix_attention_schedule="zeros",
            max_guidance_weight=5.0,
            exp_decay_alpha=1.0,
        ),
    })

    # ============================================================
    # Run each experiment
    # ============================================================
    for exp_idx, exp in enumerate(experiments):
        exp_name = exp["name"]
        method = exp["method"]
        exp_dir = pathlib.Path(output_dir) / exp_name
        results_file = exp_dir / "results.csv"

        # Skip if results already exist
        if results_file.exists():
            print(f"[SKIP] {exp_name}: results already exist at {results_file}")
            continue

        print(f"\n{'='*60}")
        print(f"[{exp_idx+1}/{len(experiments)}] Running experiment: {exp_name}")
        print(f"  method: {method}")
        print(f"  delay={inference_delay}, horizon={execute_horizon}")
        print(f"  output: {exp_dir}")
        print(f"{'='*60}\n")

        results = collections.defaultdict(list)
        
        # Run in chunks to prevent OOM
        num_batches = math.ceil(config.num_evals / config.eval_batch_size)
        
        for batch_idx in range(num_batches):
            current_batch_size = min(config.eval_batch_size, config.num_evals - batch_idx * config.eval_batch_size)
            print(f"  -> Batch {batch_idx+1}/{num_batches} (size={current_batch_size})")
            
            eval_config = _eval.EvalConfig(
                step=config.step,
                num_evals=current_batch_size,
                num_flow_steps=config.num_flow_steps,
                inference_delay=inference_delay,
                execute_horizon=execute_horizon,
                method=method,
                model=config.model,
            )
            
            # Split rngs for this chunk so different chunks get different seeds
            chunk_rngs = jax.random.split(jax.random.fold_in(rngs[0], batch_idx), len(level_paths))

            out = jax.device_get(_eval_fn(eval_config, chunk_rngs, levels, state_dicts, weak_state_dicts))
            for i in range(len(level_paths)):
                for k, v in out.items():
                    results[k].append(v[i])
                results["delay"].append(inference_delay)
                results["execute_horizon"].append(execute_horizon)
                results["method"].append(exp_name)
                results["level"].append(level_paths[i])
                results["batch_idx"].append(batch_idx)

        exp_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(results)
        df.to_csv(results_file, index=False)
        print(f"  -> Saved results to {results_file} ({len(df)} rows)")

        # Print summary
        if "returned_episode_solved" in df.columns:
            mean_solve = df["returned_episode_solved"].mean()
            print(f"  -> Mean solve rate: {mean_solve:.4f}")

    print(f"\n{'='*60}")
    print(f"All experiments complete! Results in: {output_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    tyro.cli(main)
