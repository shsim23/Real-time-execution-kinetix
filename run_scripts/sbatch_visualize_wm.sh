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

# h1 per-task checkpoints
declare -A H1_RUNS=(
  [worlds_l_grasp_easy]=unique-shape-3
  [worlds_l_catapult]=ethereal-puddle-6
  [worlds_l_cartpole_thrust]=helpful-oath-8
  [worlds_l_hard_lunar_lander]=dauntless-tree-11
  [worlds_l_mjc_half_cheetah]=atomic-brook-13
  [worlds_l_mjc_swimmer]=royal-sun-16
  [worlds_l_mjc_walker]=zany-dream-18
  [worlds_l_h17_unicycle]=feasible-frog-20
  [worlds_l_chain_lander]=comfy-donkey-23
  [worlds_l_catcher_v3]=woven-bird-24
  [worlds_l_trampoline]=smooth-paper-25
  [worlds_l_car_launch]=wobbly-planet-26
)

# h3 per-task checkpoints
declare -A H3_RUNS=(
  [worlds_l_grasp_easy]=leafy-yogurt-3
  [worlds_l_catapult]=clean-field-5
  [worlds_l_cartpole_thrust]=treasured-river-7
  [worlds_l_hard_lunar_lander]=sandy-valley-9
  [worlds_l_mjc_half_cheetah]=dry-plant-10
  [worlds_l_mjc_swimmer]=whole-firefly-12
  [worlds_l_mjc_walker]=frosty-firefly-14
  [worlds_l_h17_unicycle]=spring-tree-15
  [worlds_l_chain_lander]=leafy-water-17
  [worlds_l_catcher_v3]=dulcet-snowball-19
  [worlds_l_trampoline]=glowing-wind-21
  [worlds_l_car_launch]=fast-dawn-22
)

echo "=== Per-task h1 (history_size=1) ==="
for TASK in "${!H1_RUNS[@]}"; do
  RUN=${H1_RUNS[$TASK]}
  echo "--- $TASK (run=$RUN) ---"
  uv run src/visualize_world_model.py \
    --config.checkpoint-dir "logs-wm/${RUN}/99" \
    --config.data-path "data-pixel/${TASK}.npz" \
    --config.output-dir "wm_vis/per_task_h1/${TASK}" \
    --config.num-examples 4 \
    --config.rollout-steps 30
done

echo ""
echo "=== Per-task h3 (history_size=3) ==="
for TASK in "${!H3_RUNS[@]}"; do
  RUN=${H3_RUNS[$TASK]}
  echo "--- $TASK (run=$RUN) ---"
  uv run src/visualize_world_model.py \
    --config.checkpoint-dir "logs-wm/${RUN}/99" \
    --config.data-path "data-pixel/${TASK}.npz" \
    --config.output-dir "wm_vis/per_task_h3/${TASK}" \
    --config.num-examples 4 \
    --config.rollout-steps 30
done

echo "ALL VIS DONE"
