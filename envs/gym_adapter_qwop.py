# gym rapper

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from envs.qwop_env import QWOPEnv, ACTIONS

class QWOPGymEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, frame_stack: int = 4, debug_ocr: bool = False):
        super().__init__()
        self._env = QWOPEnv(debug_ocr=debug_ocr, frame_stack=frame_stack)
        # reset 한 번 해서 관측 모양 파악
        obs = self._env.reset()  # (stack, H, W), uint8 예상
        assert obs.ndim == 3, f"expected (stack,H,W), got {obs.shape}"
        self._obs_shape = obs.shape
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Box(
            low=0, high=255, shape=self._obs_shape, dtype=np.uint8
        )
        self._last_obs = obs

    def reset(self, seed=None, options=None):
        if seed is not None:
            # 외부 환경(브라우저 캡처)라 시드 영향은 제한적
            np.random.seed(seed)
        obs = self._env.reset()
        self._last_obs = obs
        info = {}
        return obs, info

    def step(self, action):
        obs, reward, done, info = self._env.step(int(action))
        term = bool(done)
        trunc = False  # 필요시 타임리밋 등으로 True 처리
        self._last_obs = obs
        return obs, reward, term, trunc, info

    def close(self):
        try:
            self._env.driver.quit()
        except Exception:
            pass
