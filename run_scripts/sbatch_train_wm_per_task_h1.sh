#!/bin/bash
#SBATCH --job-name=wm-pt-h1
#SBATCH --output=/home/prehj/slurm-logs/wm-pt-h1-%j.out
#SBATCH --error=/home/prehj/slurm-logs/wm-pt-h1-%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=32G
#SBATCH --time=48:00:00

export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1

cd /mnt/lustre/slurm/users/prehj/rtc_workspace/Real-time-execution-kinetix

TASKS=(
  worlds/l/grasp_easy.json
  worlds/l/catapult.json
  worlds/l/cartpole_thrust.json
  worlds/l/hard_lunar_lander.json
  worlds/l/mjc_half_cheetah.json
  worlds/l/mjc_swimmer.json
  worlds/l/mjc_walker.json
  worlds/l/h17_unicycle.json
  worlds/l/chain_lander.json
  worlds/l/catcher_v3.json
  worlds/l/trampoline.json
  worlds/l/car_launch.json
)

for TASK in "${TASKS[@]}"; do
  TASK_NAME=$(echo "$TASK" | sed 's|/|_|g; s|\.json||')
  echo "========================================"
  echo "Training: $TASK_NAME (history_size=1)"
  echo "========================================"
  uv run src/train_world_model.py \
    --config.data-dir data-pixel \
    --config.level-paths "$TASK" \
    --config.num-epochs 100 \
    --config.batch-size 64 \
    --config.save-every 10 \
    --config.model.history-size 1 \
    --config.model.num-preds 1
done

echo "ALL TASKS DONE (h1)"
