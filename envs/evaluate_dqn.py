import torch
import os
import sys
import torch.nn as nn
import numpy as np
from collections import deque, namedtuple
import random
from time import time
import time as time_module

# 경로 설정을 위해 상위 디렉토리를 sys.path에 추가 (train_dqn.py와 동일)
# [주의]: 실제 파일 경로에 맞게 수정하세요.
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# QWOPEnv 및 ACTIONS를 임포트 (qwop_env.py 파일에 따라 경로 수정 필요)
from qwop_env import QWOPEnv, ACTIONS


# ----------------------------------------------------
# 1. Q-Network 모델 정의 (QWOP_QNetwork)
#    - train_dqn.py 파일의 클래스와 동일해야 합니다.
# ----------------------------------------------------
# (QWOP_QNetwork, _get_conv_output, Transition, ReplayBuffer 클래스는
#  train_dqn.py 파일에서 그대로 복사하여 여기에 붙여넣거나,
#  별도의 model.py 파일에서 import 해야 합니다. 여기서는 편의상 생략합니다.)

class QWOP_QNetwork(nn.Module):
    """
    QWOP 이미지 상태를 처리하기 위한 CNN 모델.
    입력: (Batch_Size, Frame_Stack, Height, Width)
    """

    def __init__(self, state_shape, action_size):
        super(QWOP_QNetwork, self).__init__()
        # 상태 차원: (frame_stack, H, W) -> (4, 250, 250) (대략)
        C, H, W = state_shape

        # QWOP 이미지는 흑백이므로 (C=frame_stack)
        self.conv = nn.Sequential(
            # 입력 채널=프레임 스택(예: 4), 출력 채널=32, 커널 크기=8, 스트라이드=4
            nn.Conv2d(C, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            # 출력 채널=64, 커널 크기=4, 스트라이드=2
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            # 출력 채널=64, 커널 크기=3, 스트라이드=1
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # Conv 레이어의 출력 차원 계산 (QWOP 환경의 축소된 크기에 따라 달라짐)
        # 예시: (250x250) -> 8x8 (대략적인 값)
        # 실제 환경의 출력 크기를 계산하여 정확한 `fc_input_size`를 사용해야 함.
        # 여기서는 임시 값 1600을 사용하거나, 동적으로 계산하는 로직이 필요합니다.
        # 여기서는 Flatten 후 크기를 미리 정의했다고 가정합니다.
        fc_input_size = self._get_conv_output((C, H, W))

        self.fc = nn.Sequential(
            nn.Linear(fc_input_size, 512),
            nn.ReLU(),
            nn.Linear(512, action_size)  # 출력은 16개 행동에 대한 Q-값
        )

    # Conv 레이어 출력 크기 계산을 위한 헬퍼 함수
    def _get_conv_output(self, shape):
        # 더미 데이터로 크기 계산 (정확한 구현을 위해 필요)
        input = torch.rand(1, *shape)
        output_feat = self.conv(input)
        # Flatten된 벡터의 크기를 반환
        return int(np.prod(output_feat.size()[1:]))

    def forward(self, x):
        # QWOP 환경의 상태는 (T, H, W) 형태의 NumPy 배열입니다.
        # Pytorch CNN 입력은 (N, C, H, W)여야 합니다.
        # N=Batch Size, C=Channels (Frame Stack)
        x = self.conv(x / 255.0)  # 픽셀 값을 [0, 1]로 정규화
        x = x.reshape(x.size(0), -1)  # Flatten
        return self.fc(x)

        pass



Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """새로운 경험을 버퍼에 저장합니다."""
        self.buffer.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size):
        """버퍼에서 무작위로 배치 크기만큼의 경험을 샘플링합니다."""
        experiences = random.sample(self.buffer, batch_size)

        # NumPy 배열 경험을 Pytorch 텐서로 변환
        states = torch.as_tensor(np.array([e.state for e in experiences]), dtype=torch.float)
        actions = torch.as_tensor(np.array([e.action for e in experiences]), dtype=torch.long).unsqueeze(-1)
        rewards = torch.as_tensor(np.array([e.reward for e in experiences]), dtype=torch.float).unsqueeze(-1)
        next_states = torch.as_tensor(np.array([e.next_state for e in experiences]), dtype=torch.float)
        dones = torch.as_tensor(np.array([e.done for e in experiences]), dtype=torch.float).unsqueeze(-1)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


# ----------------------------------------------------
# 2. 평가 에이전트 (Evaluation Agent)
# ----------------------------------------------------
class EvaluationAgent:
    def __init__(self, env, state_shape, action_size, model_path):
        self.env = env
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 네트워크 초기화 및 저장된 가중치 로드
        self.policy_net = QWOP_QNetwork(state_shape, action_size).to(self.device)
        self.load_model(model_path)
        self.policy_net.eval()  # 평가 모드 설정 (Dropout, BatchNorm 비활성화)

    def load_model(self, path):
        """저장된 모델 가중치를 로드합니다."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"모델 파일이 지정된 경로에 없습니다: {path}")

        # load_state_dict를 사용하여 가중치 로드
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        print(f"✅ 모델 가중치 '{path}' 로드 완료.")

    def select_action(self, state):
        """순수한 탐욕 정책 (Greedy Policy)으로 행동을 선택합니다 (epsilon=0)."""
        with torch.no_grad():
            # 상태를 텐서로 변환하고 배치 차원 추가: (T, H, W) -> (1, T, H, W)
            state_tensor = torch.as_tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)

            # 네트워크 실행 및 최대 Q-값을 가진 행동 선택
            q_values = self.policy_net(state_tensor)
            return q_values.argmax(dim=1).item()

    def evaluate(self, num_episodes):
        """모델을 사용하여 QWOP를 플레이하며 성능을 평가합니다."""

        print("\n==================== 평가 시작 ====================")

        total_steps = 0
        total_distance = 0.0

        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0.0
            episode_steps = 0

            while True:
                # 1. 탐욕 행동 선택
                action = self.select_action(state)

                # 2. 환경과 상호작용
                next_state, reward, done, info = self.env.step(action)

                # 3. 상태 및 보상 업데이트
                state = next_state
                episode_reward += reward
                episode_steps += 1
                total_steps += 1

                dist = info.get('distance', float('nan'))

                if episode_steps % 50 == 0 or done:
                    print(
                        f"Ep {episode + 1:03d} | Step {episode_steps:04d} | Dist: {dist:.2f}m | Action: {action:02d} | Done: {done}")

                if done:
                    final_dist = info.get('distance', 0.0)
                    total_distance += final_dist

                    print(f"🎉 EPISODE {episode + 1:03d} 완료! 최종 거리: {final_dist:.2f}m, 총 스텝: {episode_steps}")
                    break

        avg_distance = total_distance / num_episodes if num_episodes > 0 else 0
        print("\n==================== 평가 결과 ====================")
        print(f"총 에피소드: {num_episodes}회")
        print(f"평균 도달 거리: {avg_distance:.2f}m")
        print("===================================================")


# ----------------------------------------------------
# 3. 실행 블록
# ----------------------------------------------------
if __name__ == '__main__':
    MODEL_FILE = "qwop_dqn_policy_net.pth"
    FRAME_STACK = 4
    NUM_EVAL_EPISODES = 10

    # 1. 환경 초기화
    # 평가 시에도 학습과 동일한 frame_stack으로 초기화해야 합니다.
    # debug_ocr을 True로 설정하면 OCR 결과 확인에 도움이 됩니다.
    env = QWOPEnv(frame_stack=FRAME_STACK, debug_ocr=False)

    # 2. 상태 및 행동 공간 정의 (reset을 통해 실제 크기 확인)
    initial_obs = env.reset()
    STATE_SHAPE = initial_obs.shape
    ACTION_SIZE = len(ACTIONS)

    # 3. 평가 에이전트 초기화 및 모델 로드
    try:
        agent = EvaluationAgent(
            env=env,
            state_shape=STATE_SHAPE,
            action_size=ACTION_SIZE,
            model_path=MODEL_FILE
        )

        # 4. 평가 실행
        agent.evaluate(num_episodes=NUM_EVAL_EPISODES)

    except FileNotFoundError as e:
        print(f"오류: {e}")
        print("💡 학습 스크립트가 실행된 디렉토리와 현재 디렉토리가 같은지 확인하거나, 모델 경로를 확인하세요.")

    finally:
        # 환경 정리
        env.close()