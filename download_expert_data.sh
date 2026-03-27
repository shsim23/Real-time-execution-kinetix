#!/bin/bash

# 다운로드할 디렉토리 생성
mkdir -p ./logs-expert/data

# 12개 Task 리스트
TASKS=(
  "grasp_easy"
  "catapult"
  "cartpole_thrust"
  "hard_lunar_lander"
  "mjc_half_cheetah"
  "mjc_swimmer"
  "mjc_walker"
  "h17_unicycle"
  "chain_lander"
  "catcher_v3"
  "trampoline"
  "car_launch"
)

echo "Downloading expert data for 12 Kinetix tasks (approx 150-200MB each)..."

# 모든 Task에 대해 병렬 다운로드 실행
for TASK in "${TASKS[@]}"; do
  echo "Downloading ${TASK}.npz..."
  # -m 옵션을 주면 전체 속도는 상승하지만, 개별 파일 내부 병렬 처리는 안될 수 있으므로 백그라운드(&) 실행
  gsutil cp gs://rtc-assets/expert/data/worlds_l_${TASK}.npz ./logs-expert/data/${TASK}.npz &
done


# 백그라운드 프로세스들이 모두 끝날 때까지 대기
wait

echo "All 12 datasets have been successfully downloaded to ./logs-expert/data/!"
