"""
Full-sync inference evaluation: generates a FRESH action chunk at every single
environment step, then executes only the first action.  This gives the model
maximum reactivity — every action is conditioned on the most recent observation.

Environment initial states are matched to the GT dataset (same RNG tree).
"""

import dataclasses
import functools
import math
import pathlib
import pickle
from typing import Sequence

import numpy as np
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import kinetix.environment.env as kenv
import kinetix.environment.env_state as kenv_state
import kinetix.environment.wrappers as wrappers
import tqdm_loggable.auto as tqdm
import tyro

import eval_flow as _eval
import model as _model
import train_expert


@dataclasses.dataclass
class Config:
    run_path: str
    output_dir: str = "matched_inference_fullsync"
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
    num_evals: int = 128  # MUST BE 128 to match GT batch size!

    # Model params
    step: int = 15000
    num_flow_steps: int = 5
    model: _model.ModelConfig = dataclasses.field(default_factory=_model.ModelConfig)


def main(config: Config):
    static_env_params = kenv_state.StaticEnvParams(**train_expert.LARGE_ENV_PARAMS, frame_skip=train_expert.FRAME_SKIP)
    env_params = kenv_state.EnvParams()
    levels = train_expert.load_levels(config.level_paths, static_env_params, env_params)
    env = kenv.make_kinetix_env_from_name("Kinetix-Symbolic-Continuous-v1", static_env_params=static_env_params)
    env = wrappers.LogWrapper(wrappers.AutoReplayWrapper(train_expert.NoisyActionWrapper(env)))

    obs_dim = jax.eval_shape(env.reset_to_level, jax.random.key(0), jax.tree.map(lambda x: x[0], levels), env_params)[0].shape[-1]
    action_dim = env.action_space(env_params).shape[0]

    # ------------------------------------------------------------------
    # Full-sync eval: fresh action chunk every step, execute only action[0]
    # ------------------------------------------------------------------
    def make_fullsync_eval_fn():
        @functools.partial(jax.jit, static_argnums=(0,))
        @functools.partial(jax.vmap, in_axes=(None, 0, 0, 0))
        def eval_fn(num_flow_steps: int, rng: jax.Array, level: kenv_state.EnvState, state_dict: dict):
            rng, key1 = jax.random.split(rng)
            rng, key_policy = jax.random.split(rng)

            @jax.vmap
            def eval_single(reset_rng, policy_rng, level):
                policy = _model.FlowPolicy(
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                    config=config.model,
                    rngs=nnx.Rngs(policy_rng),
                )
                graphdef, state = nnx.split(policy)
                state.replace_by_pure_dict(state_dict)
                policy = nnx.merge(graphdef, state)

                def step_fn(carry, _):
                    rng, obs, env_state = carry

                    # Generate a FRESH action chunk from the current observation
                    rng, key_action, key_env = jax.random.split(rng, 3)
                    action_chunk = policy.action(key_action, obs[None], num_flow_steps)[0]

                    # Execute only the FIRST action
                    action = action_chunk[0]
                    next_obs, next_env_state, reward, done, info = env.step(key_env, env_state, action, env_params)

                    return (rng, next_obs, next_env_state), (done, info, obs, action)

                obs, env_state = env.reset_to_level(reset_rng, level, env_params)
                scan_length = env_params.max_timesteps

                _, (dones, infos, obs_seq, actions) = jax.lax.scan(
                    step_fn,
                    (policy_rng, obs, env_state),
                    None,
                    length=scan_length,
                )

                return obs_seq, actions, dones, infos["returned_episode_solved"]

            eval_rngs_reset = jax.random.split(key1, config.num_evals)
            eval_rngs_policy = jax.random.split(key_policy, config.num_evals)

            return eval_single(
                eval_rngs_reset,
                eval_rngs_policy,
                jax.tree.map(lambda x: jnp.broadcast_to(x, (config.num_evals, *x.shape)), level),
            )

        return eval_fn

    # Load per-level policies
    state_dicts_list = []
    for level_path in config.level_paths:
        level_name = level_path.replace("/", "_").replace(".json", "")
        log_dirs = list(filter(lambda p: p.is_dir() and p.name.isdigit(), pathlib.Path(config.run_path).iterdir()))
        log_dirs = sorted(log_dirs, key=lambda p: int(p.name))

        if len(log_dirs) > 0:
            target_dir = log_dirs[-1]
        else:
            target_dir = pathlib.Path(config.run_path)

        policy_file = target_dir / "policies" / f"{level_name}.pkl"
        with policy_file.open("rb") as f:
            state_dicts_list.append(pickle.load(f))
    state_dicts = jax.device_put(jax.tree.map(lambda *x: jnp.array(x), *state_dicts_list))

    eval_fn = make_fullsync_eval_fn()

    print("\n======================\nFull-sync evaluation (fresh chunk every step)\n======================")

    rngs = jax.random.split(jax.random.key(config.seed), len(config.level_paths))

    out_obs, out_actions, out_dones, out_solved = eval_fn(config.num_flow_steps, rngs, levels, state_dicts)

    for i, level_path in enumerate(config.level_paths):
        level_name = level_path.replace("/", "_").replace(".json", "")

        task_obs = out_obs[i]
        task_act = out_actions[i]
        task_done = out_dones[i]
        task_solved = out_solved[i]

        episodes_obs = []
        episodes_action = []
        episodes_solved = []
        episodes_length = []

        for env_idx in range(config.num_evals):
            done_indices = np.where(task_done[env_idx, :])[0]
            if len(done_indices) > 0:
                end_idx = done_indices[0]
                episodes_obs.append(task_obs[env_idx, 0:end_idx + 1])
                episodes_action.append(task_act[env_idx, 0:end_idx + 1])
                episodes_solved.append(task_solved[env_idx, end_idx])
                episodes_length.append(end_idx + 1)

        if len(episodes_length) == 0:
            print(f"[{level_name}] NO episodes collected.")
            continue

        max_len = max(episodes_length)
        padded_obs = np.zeros((len(episodes_length), max_len, *task_obs.shape[2:]), dtype=np.float32)
        padded_act = np.zeros((len(episodes_length), max_len, *task_act.shape[2:]), dtype=np.float32)

        for j in range(len(episodes_length)):
            seq_len = episodes_length[j]
            padded_obs[j, :seq_len] = episodes_obs[j]
            padded_act[j, :seq_len] = episodes_action[j]

        sliced_data = {
            'obs': padded_obs,
            'action': padded_act,
            'solved': np.array(episodes_solved),
            'length': np.array(episodes_length),
        }

        out_folder = pathlib.Path(config.output_dir) / "fullsync"
        out_folder.mkdir(parents=True, exist_ok=True)
        np.savez(out_folder / f"{level_name}.npz", **sliced_data)

        solve_rate = np.mean(np.array(episodes_solved))
        print(f"  {level_name}: {len(episodes_length)} episodes, solve rate {solve_rate:.3f}")


if __name__ == "__main__":
    tyro.cli(main)
