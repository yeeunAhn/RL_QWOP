import torch
import os
import sys
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque, namedtuple
import random
from time import time

# 이제 'RL_QWOP/'를 기준으로 'envs' 패키지를 찾습니다.
from qwop_env import QWOPEnv, ACTIONS

# ----------------------------------------------------
# 1. DQN Q-Network 모델 정의 (CNN)
# ----------------------------------------------------
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


# ----------------------------------------------------
# 2. 경험 재생 버퍼 (Replay Buffer)
# ----------------------------------------------------
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
# 3. DQN 에이전트 (Agent)
# ----------------------------------------------------
class DQNAgent:
    def __init__(self, env, state_shape, action_size, **kwargs):
        self.env = env
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 하이퍼파라미터
        self.GAMMA = kwargs.get('gamma', 0.99)
        self.LR = kwargs.get('lr', 1e-4)
        self.BATCH_SIZE = kwargs.get('batch_size', 32)
        self.TARGET_UPDATE = kwargs.get('target_update', 1000)  # 타겟 네트워크 업데이트 주기
        self.EPSILON_START = kwargs.get('eps_start', 1.0)
        self.EPSILON_END = kwargs.get('eps_end', 0.01)
        self.EPSILON_DECAY = kwargs.get('eps_decay', 50000)
        self.REPLAY_CAPACITY = kwargs.get('replay_capacity', 10000)
        self.MIN_REPLAY_SIZE = kwargs.get('min_replay_size', 5000)

        # 네트워크 및 버퍼 초기화
        self.policy_net = QWOP_QNetwork(state_shape, action_size).to(self.device)
        self.target_net = QWOP_QNetwork(state_shape, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # 타겟 네트워크는 학습하지 않음

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.LR)
        self.memory = ReplayBuffer(self.REPLAY_CAPACITY)
        self.step_count = 0

    def select_action(self, state):
        """Epsilon-Greedy 정책에 따라 행동을 선택합니다."""
        epsilon = self.EPSILON_END + (self.EPSILON_START - self.EPSILON_END) * \
                  np.exp(-self.step_count / self.EPSILON_DECAY)

        if random.random() < epsilon:
            return random.randrange(self.action_size)  # 탐험 (랜덤 행동)
        else:
            with torch.no_grad():
                state_tensor = torch.as_tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)

                q_values = self.policy_net(state_tensor)
                return q_values.argmax(dim=1).item()  # 활용 (최대 Q-값 행동)

    def learn(self):
        """버퍼에서 샘플링하여 정책 네트워크를 학습합니다."""
        if len(self.memory) < self.MIN_REPLAY_SIZE:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(self.BATCH_SIZE)

        # 모든 텐서를 장치로 이동
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # 1. 현재 Q 값 Q(s, a) 계산 (Policy Network)
        # policy_net(states)는 모든 행동에 대한 Q-값을 반환합니다.
        # gather(1, actions)는 실제로 선택된 행동에 대한 Q-값만 추출합니다.
        current_q = self.policy_net(states).gather(1, actions)

        # 2. 타겟 Q 값 $R + \gamma \cdot \max_{a'} Q_{target}(s', a')$ 계산
        # 다음 상태에서의 최대 Q-값 $\max Q_{target}(s', a')$
        next_q_target = self.target_net(next_states).max(1)[0].unsqueeze(1)
        # 종료(done) 상태가 아닌 경우에만 다음 Q-값을 더합니다.
        target_q = rewards + (1 - dones) * self.GAMMA * next_q_target

        # Huber Loss를 사용하여 손실 계산 (DQN에서 일반적으로 사용)
        loss = nn.functional.smooth_l1_loss(current_q, target_q)

        # 3. 네트워크 업데이트
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient Clipping (선택 사항이지만 안정화에 도움)
        # for param in self.policy_net.parameters():
        #     param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def update_target_network(self):
        """일정 주기로 타겟 네트워크를 정책 네트워크로 업데이트합니다."""
        if self.step_count % self.TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def train(self, num_episodes):
        """메인 학습 루프"""

        # QWOPEnv의 frame_stack을 확인하고 설정해야 합니다.
        frame_stack = self.env.frame_stack

        for episode in range(num_episodes):
            state = self.env.reset()
            # reset()이 반환하는 상태는 (T, H, W) 형태여야 함.

            episode_start = time()
            episode_reward = 0.0

            while True:
                # 1. 행동 선택
                action = self.select_action(state)

                # 2. 환경과 상호작용
                next_state, reward, done, info = self.env.step(action)

                # 3. 경험 버퍼에 저장
                self.memory.push(state, action, reward, next_state, done)

                # 4. 상태 및 보상 업데이트
                state = next_state
                episode_reward += reward
                self.step_count += 1

                # 5. 학습 및 타겟 업데이트
                self.learn()
                self.update_target_network()

                if done:
                    print(
                        f"Ep: {episode + 1:04d} | Steps: {self.step_count} | Reward: {episode_reward:.2f} | Distance: {info.get('distance', float('nan')):.2f}m | Time: {time() - episode_start:.1f}s")
                    break

        print("DQN 학습 완료.")
        self.save_model("qwop_dqn_policy_net.pth")  # 학습 완료 후 저장

    def save_model(self, path: str):
        """정책 네트워크의 가중치를 파일로 저장합니다."""
        # 저장할 디렉토리가 없으면 생성
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)

        # state_dict 저장 (일반적으로 권장되는 방법)
        torch.save(self.policy_net.state_dict(), path)
        print(f"모델 가중치가 '{path}'에 저장되었습니다.")


# ----------------------------------------------------
# 4. 실행 예시
# ----------------------------------------------------
if __name__ == '__main__':
    # QWOP 환경 초기화 (frame_stack = 4 권장)
    FRAME_STACK = 4
    env = QWOPEnv(frame_stack=FRAME_STACK)

    # reset을 통해 상태의 실제 크기를 확인
    initial_obs = env.reset()

    print(f"--- 환경 정보 ---")
    print(f"QWOPEnv.frame_stack: {FRAME_STACK}")
    print(f"initial_obs.shape (환경에서 반환된 상태 크기): {initial_obs.shape}")
    print(f"예상 CNN 입력 크기 (Batch=1, C=T, H, W): (1, {FRAME_STACK}, H, W)")
    print("------------------")
    # (T, H, W) 형태. T=FRAME_STACK

    # 상태 및 행동 공간 정의
    STATE_SHAPE = initial_obs.shape
    ACTION_SIZE = len(ACTIONS)  # 16개 (0~15)

    # DQN 에이전트 초기화
    agent = DQNAgent(
        env=env,
        state_shape=STATE_SHAPE,
        action_size=ACTION_SIZE,
        lr=1e-4,  # 학습률
        gamma=0.99,  # 할인율
        batch_size=32,  # 배치 크기
        target_update=1000,  # 타겟 업데이트 주기
        eps_decay=50000,  # Epsilon 감소 속도
        min_replay_size=5000  # 최소 학습 시작 버퍼 크기
    )

    # 학습 시작
    print(f"DQN 학습 시작. 상태 크기: {STATE_SHAPE}, 행동 크기: {ACTION_SIZE}")
    agent.train(num_episodes=5000)
    agent.train(num_episodes=5000)  # 5000 에피소드 학습

    # 환경 종료
    # env.close() # QWOPEnv에 close 메서드가 있다면 사용