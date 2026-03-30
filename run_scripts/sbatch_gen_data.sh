#!/bin/bash
#SBATCH --job-name=gen-pixel-data
#SBATCH --output=/home/prehj/slurm-logs/gen-pixel-%j.out
#SBATCH --error=/home/prehj/slurm-logs/gen-pixel-%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=32G
#SBATCH --time=4:00:00

export PATH="$HOME/.local/bin:$PATH"

cd /mnt/lustre/slurm/users/prehj/rtc_workspace/Real-time-execution-kinetix

uv run src/generate_pixel_data.py \
    --config.run-path ./logs-expert \
    --config.num-steps 100000 \
    --config.num-envs 32 \
    --config.batch-size 64 \
    --config.pixel-size 64 \
    --config.output-dir data-pixel
