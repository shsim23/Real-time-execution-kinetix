"""
Generates inference evaluation data (actions, obs, solved) perfectly matching the 
initial states of the GT dataset.
"""

import dataclasses
import functools
import json
import math
import pathlib
import pickle
from typing import Sequence
import os

import numpy as np
from flax import struct
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
    output_dir: str = "matched_inference"
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
    inference_delay: int = 2
    execute_horizon: int = 3
    model: _model.ModelConfig = dataclasses.field(default_factory=_model.ModelConfig)

def main(config: Config):
    static_env_params = kenv_state.StaticEnvParams(**train_expert.LARGE_ENV_PARAMS, frame_skip=train_expert.FRAME_SKIP)
    env_params = kenv_state.EnvParams()
    levels = train_expert.load_levels(config.level_paths, static_env_params, env_params)
    env = kenv.make_kinetix_env_from_name("Kinetix-Symbolic-Continuous-v1", static_env_params=static_env_params)
    
    # Kinetix base env raises NotImplementedError on .step(), requires wrappers
    env = wrappers.LogWrapper(wrappers.AutoReplayWrapper(train_expert.NoisyActionWrapper(env)))

    # -------------------------------------------------------------
    # CUSTOM EVALUATION WRAPPER TO PERFECTLY MATCH EXPERT RNG TREES
    # -------------------------------------------------------------
    def make_matched_eval_fn():
        @functools.partial(jax.jit, static_argnums=(0,))
        @functools.partial(jax.vmap, in_axes=(None, 0, 0, 0, 0))
        def eval_fn(eval_config: _eval.EvalConfig, rng: jax.Array, level: kenv_state.EnvState, state_dict: dict, weak_state_dict: dict):
            # This is `rng_level` from GT loop
            rng, key1 = jax.random.split(rng)
            
            # Additional key for policy so we don't mess up the environment reset keys
            rng, key_policy = jax.random.split(rng)

            @jax.vmap
            def eval_single(reset_rng, policy_rng, level):
                policy = _model.FlowPolicy(
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                    config=eval_config.model,
                    rngs=nnx.Rngs(policy_rng),
                )
                graphdef, state = nnx.split(policy)
                state.replace_by_pure_dict(state_dict)
                policy = nnx.merge(graphdef, state)

                def execute_chunk(carry, _):
                    rng, obs, env_state, action_chunk, n = carry
                    
                    rng, key = jax.random.split(rng)
                    prefix_attention_horizon = policy.action_chunk_size - eval_config.execute_horizon
                    next_action_chunk = jax.lax.cond(
                        eval_config.method.prefix_attention_schedule == "zeros",
                        lambda: policy.action(key, obs[None], eval_config.num_flow_steps)[0],
                        lambda: policy.realtime_action(
                            key,
                            obs[None],
                            eval_config.num_flow_steps,
                            action_chunk[None],
                            eval_config.inference_delay,
                            prefix_attention_horizon,
                            eval_config.method.prefix_attention_schedule,
                            eval_config.method.max_guidance_weight,
                            eval_config.method.exp_decay_alpha,
                        )[0],
                    )

                    def step(carry, action):
                        rng, obs, env_state = carry
                        rng, key = jax.random.split(rng)
                        next_obs, next_env_state, reward, done, info = env.step(key, env_state, action, env_params)
                        return (rng, next_obs, next_env_state), (done, info, obs)

                    action_chunk_to_execute = jnp.concatenate(
                        [
                            action_chunk[: eval_config.inference_delay],
                            next_action_chunk[eval_config.inference_delay : eval_config.execute_horizon],
                        ],
                        axis=0,
                    )
                    next_action_chunk = jnp.concatenate(
                        [
                            next_action_chunk[eval_config.execute_horizon :],
                            jnp.zeros((eval_config.execute_horizon, policy.action_dim)),
                        ],
                        axis=0,
                    )
                    next_n = jnp.concatenate([n[eval_config.execute_horizon :], jnp.zeros(eval_config.execute_horizon, dtype=jnp.int32)])
                    (rng, next_obs, next_env_state), (dones, infos, obs_seq) = jax.lax.scan(
                        step, (rng, obs, env_state), action_chunk_to_execute
                    )
                    return (rng, next_obs, next_env_state, next_action_chunk, next_n), (dones, infos, obs_seq, action_chunk_to_execute)

                # MATCH: Exactly `reset_rng` passed to env.reset_to_level
                obs, env_state = env.reset_to_level(reset_rng, level, env_params)
                
                # Policy action starts with policy_rng
                action_chunk = policy.action(policy_rng, obs[None], eval_config.num_flow_steps)[0]
                n = jnp.ones(action_chunk.shape[0], dtype=jnp.int32)
                scan_length = math.ceil(env_params.max_timesteps / eval_config.execute_horizon)
                
                _, (dones, infos, obs_seq, actions) = jax.lax.scan(
                    execute_chunk,
                    (policy_rng, obs, env_state, action_chunk, n),
                    None,
                    length=scan_length,
                )
                
                dones, infos, obs_seq, actions = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), (dones, infos, obs_seq, actions))
                
                # We only need up to the FIRST done
                return obs_seq, actions, dones, infos["returned_episode_solved"]

            eval_rngs_reset = jax.random.split(key1, eval_config.num_evals)
            eval_rngs_policy = jax.random.split(key_policy, eval_config.num_evals)
            
            return eval_single(eval_rngs_reset, eval_rngs_policy, jax.tree.map(lambda x: jnp.broadcast_to(x, (eval_config.num_evals, *x.shape)), level))

        return eval_fn

    # Load policies from best checkpoints
    state_dicts_list = []
    for level_path in config.level_paths:
        level_name = level_path.replace("/", "_").replace(".json", "")
        log_dirs = list(filter(lambda p: p.is_dir() and p.name.isdigit(), pathlib.Path(config.run_path).iterdir()))
        log_dirs = sorted(log_dirs, key=lambda p: int(p.name))
        
        # If run_path is directly `logs-bc/bc/31/`, then step folder might just be `15000` inside `policies`? 
        # Actually in Kinetix, run_path like `./logs-bc/bc/31` contains `policies/`. 
        # Wait, if `run_path` is `./logs-bc/bc`, there is a single run dir `31`.
        # `log_dirs` will find `[31]`.
        if len(log_dirs) > 0:
            target_dir = log_dirs[-1] # Usually the seed folder
        else:
            target_dir = pathlib.Path(config.run_path)
            
        policy_file = target_dir / "policies" / f"{level_name}.pkl"
        with policy_file.open("rb") as f:
            state_dicts_list.append(pickle.load(f))
    state_dicts = jax.device_put(jax.tree.map(lambda *x: jnp.array(x), *state_dicts_list))
    
    weak_state_dicts = jax.tree.map(lambda x: jnp.zeros_like(x), state_dicts)
    
    obs_dim = jax.eval_shape(env.reset_to_level, jax.random.key(0), jax.tree.map(lambda x: x[0], levels), env_params)[0].shape[-1]
    action_dim = env.action_space(env_params).shape[0]

    eval_fn = make_matched_eval_fn()

    sweep_configs = [
        {"name": "alpha_1.0", "method": _eval.RealtimeMethodConfig(prefix_attention_schedule="exp", max_guidance_weight=5.0, exp_decay_alpha=1.0)},
        {"name": "alpha_3.0", "method": _eval.RealtimeMethodConfig(prefix_attention_schedule="exp", max_guidance_weight=5.0, exp_decay_alpha=3.0)},
        {"name": "alpha_5.0", "method": _eval.RealtimeMethodConfig(prefix_attention_schedule="exp", max_guidance_weight=5.0, exp_decay_alpha=5.0)},
        {"name": "hard_masking", "method": _eval.RealtimeMethodConfig(prefix_attention_schedule="zeros", max_guidance_weight=5.0, exp_decay_alpha=1.0)},
    ]

    for sweep_target in sweep_configs:
        sweep_name = sweep_target["name"]
        method = sweep_target["method"]
        print(f"\n======================\nEvaluating {sweep_name}\n======================")
        
        # Base RNG tree top must equal 0 to match GT generate_data.py
        rngs = jax.random.split(jax.random.key(config.seed), len(config.level_paths))
        
        eval_config = _eval.EvalConfig(
            step=config.step, num_evals=config.num_evals, num_flow_steps=config.num_flow_steps,
            inference_delay=config.inference_delay, execute_horizon=config.execute_horizon,
            method=method, model=config.model,
        )

        out_obs, out_actions, out_dones, out_solved = eval_fn(eval_config, rngs, levels, state_dicts, weak_state_dicts)
        
        for i, level_path in enumerate(config.level_paths):
            level_name = level_path.replace("/", "_").replace(".json", "")
            
            # Shape for this task: [128 envs, max_timesteps, ...]
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
                    episodes_obs.append(task_obs[env_idx, 0:end_idx+1])
                    episodes_action.append(task_act[env_idx, 0:end_idx+1])
                    episodes_solved.append(task_solved[env_idx, end_idx])
                    episodes_length.append(end_idx + 1)
                    
            if len(episodes_length) == 0:
                print(f"[{level_name}] NO successful episodes collected.")
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
            
            out_folder = pathlib.Path(config.output_dir) / sweep_name
            out_folder.mkdir(parents=True, exist_ok=True)
            np.savez(out_folder / f"{level_name}.npz", **sliced_data)
            
            print(f"  Saved {level_name}.npz ({len(episodes_length)} perfectly matched episodes).")

if __name__ == "__main__":
    tyro.cli(main)
