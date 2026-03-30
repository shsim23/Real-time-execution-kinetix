"""World Model 사용 예제

이 스크립트는 학습된 JEPA World Model을 로드하여
pixel observation + action sequence로부터 미래 프레임을 예측하는 방법을 보여줍니다.

실행:
    uv run src/example_world_model.py

필요 파일:
    - logs-wm/per_task_h3/h3_grasp_easy/99/  (학습된 체크포인트)
    - data-pixel/worlds_l_grasp_easy.npz       (pixel 데이터셋)
"""

import pickle
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import numpy as np

import world_model as wm

# ──────────────────────────────────────────────
# 1. 모델 로드
# ──────────────────────────────────────────────

# 체크포인트 경로 (per_task_h3/h3_<태스크명>/99 구조)
CKPT_DIR = "logs-wm/per_task_h3/h3_grasp_easy/99"

# config.pkl: 모델 구조 정보 (WorldModelConfig)
with open(f"{CKPT_DIR}/config.pkl", "rb") as f:
    config = pickle.load(f)

# world_model.pkl: 학습된 가중치
with open(f"{CKPT_DIR}/world_model.pkl", "rb") as f:
    state_dict = pickle.load(f)

# 모델 생성 후 가중치 로드
model = wm.JEPA(config, rngs=nnx.Rngs(jax.random.key(0)))
graphdef, state = nnx.split(model)
state.replace_by_pure_dict(state_dict)
model = nnx.merge(graphdef, state)

print(f"모델 로드 완료: {sum(x.size for x in jax.tree.leaves(nnx.state(model, nnx.Param))):,} params")
print(f"  history_size={config.history_size}, embed_dim={config.embed_dim}, img_size={config.img_size}")

# ──────────────────────────────────────────────
# 2. 데이터 로드
# ──────────────────────────────────────────────

DATA_PATH = "data-pixel/worlds_l_grasp_easy.npz"
data = np.load(DATA_PATH)

pixels = data["pixels"]    # (steps, envs, 64, 64, 3) uint8
actions = data["action"]   # (steps, envs, 6) float32
done = data["done"]        # (steps, envs) bool

print(f"\n데이터 로드: pixels={pixels.shape}, actions={actions.shape}")

# ──────────────────────────────────────────────
# 3. Context 준비
# ──────────────────────────────────────────────
# history_size=3이면 직전 3프레임이 context
# history_size=1이면 단일 프레임이 context

hs = config.history_size  # 3
env_idx = 0               # 첫 번째 env 사용
start = 10                # 10번째 step부터 시작
rollout_steps = 30        # 30 step 미래 예측

# context 프레임 (history_size개)
ctx_pixels = pixels[start:start + hs, env_idx]        # (3, 64, 64, 3) uint8
ctx_actions = actions[start:start + hs, env_idx]       # (3, 6) float32

# 미래 action sequence (expert가 실제로 수행한 action)
future_actions = actions[start + hs:start + hs + rollout_steps, env_idx]  # (30, 6)

# GT 미래 프레임 (비교용)
gt_future = pixels[start + hs:start + hs + rollout_steps, env_idx]  # (30, 64, 64, 3)

# JAX 텐서로 변환 (batch dim 추가, pixel은 [0,1] 정규화)
ctx_px = jnp.array(ctx_pixels[None] / 255.0)      # (1, 3, 64, 64, 3) float32
ctx_act = jnp.array(ctx_actions[None])              # (1, 3, 6) float32
fut_act = jnp.array(future_actions[None])            # (1, 30, 6) float32

# ──────────────────────────────────────────────
# 4. Encoding: pixel → embedding
# ──────────────────────────────────────────────
# encode()는 pixel과 action을 받아서 latent embedding을 반환

emb, act_emb = model.encode(ctx_px, ctx_act)
print(f"\nEncoding 결과:")
print(f"  state embedding: {emb.shape}")      # (1, 3, 192)
print(f"  action embedding: {act_emb.shape}")  # (1, 3, 192)

# ──────────────────────────────────────────────
# 5. Rollout: 미래 state embedding 예측
# ──────────────────────────────────────────────
# rollout()은 context + future actions를 받아서
# autoregressive하게 미래 embedding을 예측
# 내부적으로 ARPredictor가 sliding window로 다음 step을 예측

pred_emb = model.rollout(ctx_px, ctx_act, fut_act)
print(f"\nRollout 결과:")
print(f"  예측 embedding: {pred_emb.shape}")   # (1, 34, 192) = 3 context + 30 future + 1 final

# ──────────────────────────────────────────────
# 6. Decoding: embedding → pixel 복원
# ──────────────────────────────────────────────
# decode()는 embedding을 pixel로 변환 (decoder 필요)
# 주의: le-wm 원래 설계에는 decoder가 없음. 시각화를 위해 추가한 것

pred_pixels = model.decode(pred_emb)   # (1, 34, 64, 64, 3) float [0,1]

# numpy uint8로 변환
pred_np = np.clip(np.array(pred_pixels[0]) * 255, 0, 255).astype(np.uint8)

# pred_np[:hs]   → context 프레임 복원 (encoder→decoder 품질 확인)
# pred_np[hs:]   → 예측된 미래 프레임 (world model 예측 품질)
print(f"\nDecoding 결과:")
print(f"  예측 pixel: {pred_np.shape}")  # (34, 64, 64, 3) uint8

# ──────────────────────────────────────────────
# 7. 정량 평가: 예측 vs GT
# ──────────────────────────────────────────────

# Embedding MSE: GT 프레임을 encode한 것과 예측 embedding 비교
all_px = jnp.array(
    pixels[start:start + hs + rollout_steps, env_idx][None] / 255.0
)
all_act = jnp.array(
    actions[start:start + hs + rollout_steps, env_idx][None]
)
gt_emb, _ = model.encode(all_px, all_act)

emb_mse = float(jnp.mean((pred_emb[0, hs:hs + rollout_steps] - gt_emb[0, hs:]) ** 2))
print(f"\n정량 평가:")
print(f"  Embedding MSE (30-step): {emb_mse:.6f}")

# Pixel MSE: 예측 pixel과 GT pixel 비교
pixel_mse = float(np.mean(
    (pred_np[hs:hs + rollout_steps].astype(float) - gt_future.astype(float)) ** 2
)) / (255 ** 2)
print(f"  Pixel MSE (30-step):     {pixel_mse:.6f}")

# Reconstruction MSE: context 프레임의 encoder→decoder 품질
recon_mse = float(np.mean(
    (pred_np[:hs].astype(float) - ctx_pixels.astype(float)) ** 2
)) / (255 ** 2)
print(f"  Reconstruction MSE:      {recon_mse:.6f}")

# ──────────────────────────────────────────────
# 8. 이미지 저장 (선택)
# ──────────────────────────────────────────────

try:
    import imageio
    # 예측 미래 프레임을 gif로 저장
    imageio.mimwrite("example_predicted.gif", pred_np[hs:hs + rollout_steps], fps=8, loop=0)
    # GT 미래 프레임을 gif로 저장
    imageio.mimwrite("example_gt.gif", gt_future, fps=8, loop=0)
    print(f"\n저장 완료: example_predicted.gif, example_gt.gif")
except ImportError:
    print("\nimageio 없이도 pred_np 배열로 직접 사용 가능합니다")
