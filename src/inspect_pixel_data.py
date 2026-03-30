"""Inspect generated pixel data from generate_pixel_data.py.

Usage:
    uv run src/inspect_pixel_data.py data-pixel/worlds_l_grasp_easy.npz
    uv run src/inspect_pixel_data.py data-pixel/  # inspect all files in dir
"""

import pathlib
import sys

import imageio
import numpy as np


def inspect_file(path: pathlib.Path):
    print(f"\n{'='*60}")
    print(f"File: {path.name} ({path.stat().st_size / 1e6:.1f} MB)")
    print(f"{'='*60}")

    data = np.load(path)

    for key in data.files:
        arr = data[key]
        print(f"  {key:10s}  shape={str(arr.shape):30s}  dtype={arr.dtype}")

    pixels = data["pixels"]
    actions = data["action"]
    done = data["done"]

    num_steps, num_envs = done.shape
    total_transitions = num_steps * num_envs
    num_episodes = done.sum()
    print(f"\n  Steps: {num_steps:,} x {num_envs} envs = {total_transitions:,} transitions")
    print(f"  Episodes: {int(num_episodes):,}")
    print(f"  Pixel range: [{pixels.min()}, {pixels.max()}]")
    print(f"  Action range: [{actions.min():.3f}, {actions.max():.3f}]")

    if "solved" in data.files:
        solved = data["solved"]
        solve_rate = (solved * done).sum() / max(num_episodes, 1)
        print(f"  Solve rate: {solve_rate:.1%}")

    # Save sample frames as gif (first episode of env 0)
    sample_dir = path.parent / "samples"
    sample_dir.mkdir(exist_ok=True)

    frames = pixels[:, 0]  # (steps, H, W, 3) — env 0
    done_env0 = done[:, 0]
    first_done = np.argmax(done_env0)
    if first_done == 0 and not done_env0[0]:
        first_done = len(done_env0)  # no episode boundary
    ep_len = min(first_done + 1, 200)  # cap at 200 frames

    gif_path = sample_dir / f"{path.stem}_sample.gif"
    imageio.mimwrite(gif_path, frames[:ep_len], fps=15, loop=0)
    print(f"\n  Saved sample gif ({ep_len} frames): {gif_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: uv run src/inspect_pixel_data.py <path.npz or directory>")
        sys.exit(1)

    target = pathlib.Path(sys.argv[1])

    if target.is_dir():
        files = sorted(target.glob("*.npz"))
        if not files:
            print(f"No .npz files found in {target}")
            sys.exit(1)
        for f in files:
            inspect_file(f)
    else:
        inspect_file(target)


if __name__ == "__main__":
    main()
