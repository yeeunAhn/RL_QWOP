# eval_dqn.py  (train_dqn.py랑 같은 폴더, rl/ 안에 두는 기준)

import os
import sys
import time
import torch
import torch.nn as nn
import numpy as np

# -----------------------------
# 프로젝트 루트 및 경로 설정
# -----------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))   # rl/
project_root = os.path.dirname(script_dir)                # RL_QWOP/
sys.path.append(project_root)

# env 임포트
from envs.qwop_env import QWOPEnv, ACTIONS

# 학습 때 쓰던 Q-네트워크 임포트
from rl.train_dqn import QWOP_QNetwork


def load_model(model_path: str, state_shape, action_size, device):
    """학습된 best_model_*.pth 를 로드해서 Q 네트워크 리턴"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없음: {model_path}")

    print(f"[INFO] 모델 로드: {model_path}")
    model = QWOP_QNetwork(state_shape, action_size).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def select_action_greedy(model: nn.Module, state: np.ndarray, device):
    """탐험 없이 Q값 최대 행동 선택 (evaluation 모드)"""
    with torch.no_grad():
        s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)  # (1, C, H, W)
        q = model(s)
        action = q.argmax(dim=1).item()
    return action


def evaluate(model_path: str,
             num_episodes: int = 5,
             frame_stack: int = 2,
             background_safe: bool = True,
             render_sleep: float = 0.01):
    """학습된 모델을 여러 에피소드 동안 평가"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # 환경 생성
    env = QWOPEnv(debug_ocr=False, frame_stack=frame_stack, background_safe=background_safe)

    # 상태/행동 크기 확인
    initial_obs = env.reset()
    state_shape = initial_obs.shape          # (C, H, W)
    action_size = len(ACTIONS)
    print(f"[INFO] STATE_SHAPE: {state_shape}, ACTION_SIZE: {action_size}")

    # 모델 로드
    model = load_model(model_path, state_shape, action_size, device)

    # 에피소드 루프
    for ep in range(1, num_episodes + 1):
        state = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        start_t = time.time()

        print(f"\n================= EVALUATE EPISODE {ep:02d} =================")

        while not done:
            action = select_action_greedy(model, state, device)

            next_state, reward, done, info = env.step(action)

            total_reward += reward
            steps += 1
            state = next_state

            # 중간중간 로그
            if steps % 30 == 0:
                dist = info.get("distance", float("nan"))
                dist_str = f"{dist:.2f}m" if not np.isnan(dist) else "N/A"
                print(f"[EP {ep:02d}] Step: {steps:04d} | Distance: {dist_str} | Reward: {total_reward:.2f}")

            if render_sleep > 0:
                time.sleep(render_sleep)

        # 에피소드 종료 로그
        dist = info.get("distance", float("nan"))
        dist_str = f"{dist:.2f}m" if not np.isnan(dist) else "N/A"
        elapsed = time.time() - start_t

        print(f"\n🎉 EP {ep:02d} 종료")
        print(f"   Steps     : {steps}")
        print(f"   Distance  : {dist_str}")
        print(f"   Total Rwd : {total_reward:.2f}")
        print(f"   Time      : {elapsed:.1f}s")

    env.close()
    print("\n✅ 평가 종료")


if __name__ == "__main__":
    # 여기만 너 상황에 맞게 바꿔주면 됨
    MODEL_PATH = os.path.join(
        project_root,
        "checkpoints",
        "ver3",
        "best_model_30.10m.pth",   # 네가 말한 그 파일 이름
    )

    evaluate(
        model_path=MODEL_PATH,
        num_episodes=10,          # 몇 번 돌려볼지
        frame_stack=2,           # train_dqn.py에서 쓰던 FRAME_STACK이랑 맞추기
        background_safe=True,    # 너가 평소 쓰는 옵션
        render_sleep=0.01,       # 너무 빨리 돌면 0으로 줄여도 됨
    )
