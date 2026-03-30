#!/bin/bash
#SBATCH --job-name=wm-h3
#SBATCH --output=/home/prehj/slurm-logs/wm-h3-%j.out
#SBATCH --error=/home/prehj/slurm-logs/wm-h3-%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=32G
#SBATCH --time=48:00:00

export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1

cd /mnt/lustre/slurm/users/prehj/rtc_workspace/Real-time-execution-kinetix

uv run src/train_world_model.py \
    --config.data-dir data-pixel \
    --config.num-epochs 100 \
    --config.batch-size 64 \
    --config.save-every 10 \
    --config.model.history-size 3 \
    --config.model.num-preds 1
