#!/bin/bash
#SBATCH --job-name=wm-vis
#SBATCH --output=/home/prehj/slurm-logs/wm-vis-%j.out
#SBATCH --error=/home/prehj/slurm-logs/wm-vis-%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:a6000:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=32G
#SBATCH --time=4:00:00

export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1

cd /mnt/lustre/slurm/users/prehj/rtc_workspace/Real-time-execution-kinetix

TASKS=(
  grasp_easy
  catapult
  cartpole_thrust
  hard_lunar_lander
  mjc_half_cheetah
  mjc_swimmer
  mjc_walker
  h17_unicycle
  chain_lander
  catcher_v3
  trampoline
  car_launch
)

echo "=== Per-task h1 (history_size=1) ==="
for TASK in "${TASKS[@]}"; do
  echo "--- $TASK ---"
  uv run src/visualize_world_model.py \
    --config.checkpoint-dir "logs-wm/per_task_h1/h1_${TASK}/99" \
    --config.data-path "data-pixel/worlds_l_${TASK}.npz" \
    --config.output-dir "wm_vis/per_task_h1/worlds_l_${TASK}" \
    --config.num-examples 4 \
    --config.rollout-steps 30
done

echo ""
echo "=== Per-task h3 (history_size=3) ==="
for TASK in "${TASKS[@]}"; do
  echo "--- $TASK ---"
  uv run src/visualize_world_model.py \
    --config.checkpoint-dir "logs-wm/per_task_h3/h3_${TASK}/99" \
    --config.data-path "data-pixel/worlds_l_${TASK}.npz" \
    --config.output-dir "wm_vis/per_task_h3/worlds_l_${TASK}" \
    --config.num-examples 4 \
    --config.rollout-steps 30
done

echo "ALL VIS DONE"
