import torch
import os
import sys
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque, namedtuple
import random
from time import time
import time as time_module

# 현재 스크립트 파일(train_dqn.py)이 있는 디렉토리(rl/)의 절대 경로를 가져옵니다.
script_dir = os.path.dirname(os.path.abspath(__file__))
# 부모 디렉토리(프로젝트 루트, RL_QWOP/)의 절대 경로를 가져옵니다.
project_root = os.path.dirname(script_dir)
# 파이썬이 모듈을 찾을 수 있도록 프로젝트 루트 경로를 시스템 경로에 추가합니다.
sys.path.append(project_root)

# 💡 envs 폴더에 있는 qwop_env를 임포트합니다.
from envs.qwop_env import QWOPEnv, ACTIONS


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
        # 상태 차원: (frame_stack, H, W)
        C, H, W = state_shape

        self.conv = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # Conv 레이어의 출력 차원 동적 계산
        fc_input_size = self._get_conv_output((C, H, W))

        self.fc = nn.Sequential(
            nn.Linear(fc_input_size, 512),
            nn.ReLU(),
            nn.Linear(512, action_size)  # 출력은 행동 개수에 대한 Q-값
        )

    def _get_conv_output(self, shape):
        # 더미 데이터로 크기 계산
        with torch.no_grad():
            input = torch.rand(1, *shape)
            output_feat = self.conv(input)
            # Flatten된 벡터의 크기를 반환
            return int(np.prod(output_feat.size()[1:]))

    def forward(self, x):
        # (N, C, H, W)
        x = self.conv(x / 255.0)  # 픽셀 값을 [0, 1]로 정규화
        x = torch.flatten(x, 1)  # Batch 차원(0)을 제외하고 모두 Flatten
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

    def sample(self, batch_size, device):
        """버퍼에서 무작위로 배치 크기만큼의 경험을 샘플링합니다."""
        experiences = random.sample(self.buffer, batch_size)

        states = torch.tensor(np.array([e.state for e in experiences]), dtype=torch.float, device=device)
        actions = torch.tensor(np.array([e.action for e in experiences]), dtype=torch.long, device=device).unsqueeze(-1)
        rewards = torch.tensor(np.array([e.reward for e in experiences]), dtype=torch.float, device=device).unsqueeze(
            -1)
        next_states = torch.tensor(np.array([e.next_state for e in experiences]), dtype=torch.float, device=device)
        dones = torch.tensor(np.array([e.done for e in experiences]), dtype=torch.float, device=device).unsqueeze(-1)

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
        print(f"Device: {self.device} 사용")

        # 하이퍼파라미터
        self.GAMMA = kwargs.get('gamma', 0.99)
        self.LR = kwargs.get('lr', 1e-4)
        self.BATCH_SIZE = kwargs.get('batch_size', 32)
        self.TARGET_UPDATE = kwargs.get('target_update', 1000)  # 타겟 네트워크 업데이트 주기
        self.EPSILON_START = kwargs.get('eps_start', 1.0)
        self.EPSILON_END = kwargs.get('eps_end', 0.01)
        self.EPSILON_DECAY = kwargs.get('eps_decay', 50000)
        self.REPLAY_CAPACITY = kwargs.get('replay_capacity', 50000)
        self.MIN_REPLAY_SIZE = kwargs.get('min_replay_size', 5000)

        # 네트워크 및 버퍼 초기화
        self.policy_net = QWOP_QNetwork(state_shape, action_size).to(self.device)
        self.target_net = QWOP_QNetwork(state_shape, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # 타겟 네트워크는 학습하지 않음

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.LR)
        self.memory = ReplayBuffer(self.REPLAY_CAPACITY)
        self.step_count = 0
        self.current_epsilon = self.EPSILON_START  # 💡 초기화

    def select_action(self, state):
        """Epsilon-Greedy 정책에 따라 행동을 선택합니다."""
        epsilon = self.EPSILON_END + (self.EPSILON_START - self.EPSILON_END) * \
                  np.exp(-self.step_count / self.EPSILON_DECAY)

        self.current_epsilon = epsilon  # 로그용

        if random.random() < epsilon:
            return random.randrange(self.action_size)  # 탐험 (랜덤 행동)
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze(0)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax(dim=1).item()  # 활용 (최대 Q-값 행동)

    def learn(self):
        """버퍼에서 샘플링하여 정책 네트워크를 학습합니다."""
        if len(self.memory) < self.MIN_REPLAY_SIZE:
            return  # 버퍼가 충분히 찰 때까지 학습하지 않음

        states, actions, rewards, next_states, dones = self.memory.sample(self.BATCH_SIZE, self.device)

        # 1. 현재 Q 값 Q(s, a) 계산 (Policy Network)
        current_q = self.policy_net(states).gather(1, actions)

        # 2. 타겟 Q 값 $R + \gamma \cdot \max_{a'} Q_{target}(s', a')$ 계산
        with torch.no_grad():
            next_q_target = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (1 - dones) * self.GAMMA * next_q_target

        # Huber Loss를 사용하여 손실 계산
        loss = nn.functional.smooth_l1_loss(current_q, target_q)

        # 3. 네트워크 업데이트
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

    def update_target_network(self):
        """일정 주기로 타겟 네트워크를 정책 네트워크로 업데이트합니다."""
        if self.step_count % self.TARGET_UPDATE == 0:
            print(f"--- Step {self.step_count}: 타겟 네트워크 업데이트 ---")
            self.target_net.load_state_dict(self.policy_net.state_dict())

    # 💡 [수정] train 함수에 자동 저장을 위한 인자 추가
    def train(self, num_episodes, checkpoint_dir, checkpoint_interval=5000):
        """메인 학습 루프"""

        # 💡 자동 저장을 위해 현재 스텝 기준으로 마지막 저장 지점 계산
        last_checkpoint_step = (self.step_count // checkpoint_interval) * checkpoint_interval

        for episode in range(1, num_episodes + 1):
            state = self.env.reset()
            episode_start = time()
            episode_reward = 0.0
            episode_steps = 0

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
                episode_steps += 1

                # 5. 학습 (버퍼가 찼을 때만)
                self.learn()

                # 6. 타겟 네트워크 업데이트 (일정 주기마다)
                self.update_target_network()
                import time as t
                t.sleep(0.001)

                # 💡 [수정] 5000 스텝마다 중간 저장 로직
                current_checkpoint_step = (self.step_count // checkpoint_interval) * checkpoint_interval
                if current_checkpoint_step > last_checkpoint_step:
                    last_checkpoint_step = current_checkpoint_step  # 마지막 저장 지점 갱신

                    # ⭐️ [추가] 현재 시간 가져오기 및 포맷팅
                    timestamp = time()
                    timestamp_str = time_module.strftime("%Y%m%d_%H%M%S", time_module.localtime(timestamp))

                    # ⭐️ [수정] 파일명에 시간 문자열 포함
                    filename = f"qwop_checkpoint_{timestamp_str}_steps{last_checkpoint_step}.pth"

                    self.save_model(os.path.join(checkpoint_dir, filename))
                    print(f"\n--- 💾 중간 저장 완료: {filename} ---")
                if done:
                    dist = info.get('distance', float('nan'))
                    dist_str = f"{dist:.2f}m" if not np.isnan(dist) else "N/A"

                    print(
                        f"Ep: {episode:04d} | Total Steps: {self.step_count} "
                        f"| Ep Reward: {episode_reward:.2f} | Ep Steps: {episode_steps} "
                        f"| Distance: {dist_str} | Epsilon: {self.current_epsilon:.3f} "
                        f"| Time: {time() - episode_start:.1f}s"
                    )
                    break

        print("DQN 학습 완료.")

    def save_model(self, path: str):
        """정책 네트워크의 가중치를 파일로 저장합니다."""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save(self.policy_net.state_dict(), path)
        print(f"모델 가중치가 '{path}'에 저장되었습니다.")

    def load_model(self, path: str):
        """파일에서 정책 네트워크의 가중치를 로드합니다."""
        if not os.path.exists(path):
            print(f"경고: 모델 파일 '{path}'를 찾을 수 없습니다. 새 모델로 시작합니다.")
            return False

        try:
            self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
            self.target_net.load_state_dict(self.policy_net.state_dict())  # 타겟넷도 동기화
            print(f"✅ 모델 가중치 '{path}' 로드 완료.")
            return True
        except Exception as e:
            print(f"❌ 모델 로드 중 오류 발생: {e}. 새 모델로 시작합니다.")
            return False


# ----------------------------------------------------
# 4. 실행
# ----------------------------------------------------

if __name__ == '__main__':

    # 체크포인트 디렉토리를 프로젝트 루트 기준으로 설정
    CHECKPOINT_DIR = os.path.join(project_root, "checkpoints")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # QWOP 환경 초기화 (frame_stack = 4 권장)
    FRAME_STACK = 4
    env = QWOPEnv(debug_ocr=False, frame_stack=FRAME_STACK, background_safe=True)

    initial_obs = env.reset()

    # 상태 및 행동 공간 정의
    STATE_SHAPE = initial_obs.shape
    ACTION_SIZE = len(ACTIONS)

    # DQN 에이전트 초기화
    agent = DQNAgent(
        env=env,
        state_shape=STATE_SHAPE,
        action_size=ACTION_SIZE,
        lr=1e-4,  # 학습률
        gamma=0.99,  # 할인 계수
        batch_size=32,  # 배치 크기
        target_update=2000,  # 타겟 업데이트 주기 (스텝 기준)
        eps_decay=40000,  # Epsilon 감소 속도 (더 느리게)
        replay_capacity=20000,  # 리플레이 버퍼 크기
        min_replay_size=5000  # 최소 학습 시작 크기
    )

    # 💡 [수정] 모델 로드 로직 (이어하기 원할 때 사용)
    # -------------------------------------------------------------------
    # ⭐️ True로 바꾸고 아래 2줄 수정하면 이어하기
    LOAD_MODEL = False
    LOAD_STEP = 10000
    MODEL_TO_LOAD = "checkpoints/qwop_checkpoint_20251119_160841_steps10000.pth"  # ⭐️ 이어할 파일명

    if LOAD_MODEL and agent.load_model(MODEL_TO_LOAD):
        agent.step_count = LOAD_STEP
        print(f"✅ 모델 로드 성공. {LOAD_STEP} 스텝부터 학습을 재개합니다.")
    else:
        if LOAD_MODEL:
            print(f"❌ 모델 로드 실패. 0 스텝부터 새로 시작합니다.")
        else:
            print("ℹ️ 새 모델로 0 스텝부터 학습을 시작합니다.")
        agent.step_count = 0  # ⭐️ 0부터 새로 시작
    # -------------------------------------------------------------------

    # 학습 시작
    print(f"DQN 학습 시작. 상태 크기: {STATE_SHAPE}, 행동 크기: {ACTION_SIZE}")
    print(f"모델 저장 경로: {CHECKPOINT_DIR}")

    try:
        # 💡 [수정] train 함수에 자동 저장 경로와 간격(5000) 전달
        agent.train(num_episodes=5000,  # (넉넉하게)
                    checkpoint_dir=CHECKPOINT_DIR,
                    checkpoint_interval=5000)  # ⭐️ 5000 스텝마다 저장

        # 모든 학습이 완료된 eps_decay=100000후, 타임스탬프가 포함된 파일명으로 저장합니다.
        print("\n[학습 완료] 모든 에피소드 학습 완료. 최종 모델을 저장합니다.")
        timestamp = time()
        timestamp_str = time_module.strftime("%Y%m%d_%H%M%S", time_module.localtime(timestamp))
        filename = f"qwop_completed_{timestamp_str}_steps{agent.step_count}.pth"
        agent.save_model(os.path.join(CHECKPOINT_DIR, filename))
        print(f"\n✅ 전체 학습 완료! 모델이 {filename} 에 저장되었습니다.")

    except KeyboardInterrupt:
        # Ctrl+C 감지 시 실행
        timestamp = time()
        timestamp_str = time_module.strftime("%Y%m%d_%H%M%S", time_module.localtime(timestamp))
        filename = f"qwop_interrupted_{timestamp_str}_steps{agent.step_count}.pth"
        print(f"\n[Ctrl+C 감지] 학습 중단 요청. 모델을 {filename}에 저장합니다.")
        agent.save_model(os.path.join(CHECKPOINT_DIR, filename))

    except Exception as e:
        print(f"\n[오류 감지] 예상치 못한 오류 발생: {e}")
        # 오류 발생 시에도 저장을 시도합니다.
        timestamp = time()
        timestamp_str = time_module.strftime("%Y%m%d_%H%M%S", time_module.localtime(timestamp))
        filename = f"qwop_error_{timestamp_str}_steps{agent.step_count}.pth"
        print(f"오류 발생 전 모델을 {filename}에 저장합니다.")
        agent.save_model(os.path.join(CHECKPOINT_DIR, filename))
        # ⭐️ 오류 스택 트레이스를 출력하여 디버깅 돕기
        import traceback

        traceback.print_exc()


    finally:
        # 환경 종료 (정상 종료, Ctrl+C 종료, 오류 종료 모두 실행됨)
        if hasattr(env, 'close'):
            env.close()