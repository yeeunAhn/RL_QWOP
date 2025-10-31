# QWOPEnv_gymlike.py
from typing import Union, Tuple, Dict
from collections import deque
import numpy as np
import cv2
from time import sleep, time
from mss import mss
from PIL import Image

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from pynput.keyboard import Controller, Key

ACTIONS = {
    0: [],           # no-op
    1: ['q'],
    2: ['w'],
    3: ['o'],
    4: ['p'],
    5: ['q','w'],
    6: ['o','p'],
}

def preprocess_frame(gray: np.ndarray, out_hw: Tuple[int,int]=(80,80)) -> np.ndarray:
    # gray: (H,W) uint8 -> (80,80) float32 [0,1]
    img = cv2.resize(gray, out_hw, interpolation=cv2.INTER_AREA)
    return (img.astype(np.float32) / 255.0)

class QWOPEnv:
    """Gym 스타일 step/reset 지원하는 QWOP 환경 (Ruffle + SWF 화면 캡처 기반)"""
    def __init__(self,
                 url: str = 'http://0.0.0.0:8000/',
                 stack: int = 4,
                 action_hold_sec: float = 0.12,
                 idle_done_sec: float = 3.0,
                 step_timeout_sec: float = 30.0):
        self.stack = stack
        self.frames = deque(maxlen=stack)
        self.action_hold_sec = action_hold_sec
        self.idle_done_sec = idle_done_sec
        self.step_timeout_sec = step_timeout_sec

        # --- Browser ---
        options = webdriver.ChromeOptions()
        # options.add_argument('--headless')  # mss 캡처 정확도 위해 권장 X
        options.add_argument("--window-size=1000,1000")

        self.driver = webdriver.Chrome(options=options)
        try:
            self.driver.set_window_position(0, 0)  # mss 좌표 정합성 ↑
        except Exception:
            pass
        self.driver.get(url)  # ← 네가 요구한 그대로 유지

        # --- Keyboard ---
        self.keyboard = Controller()

        # --- Wait for Ruffle element ---
        wait = WebDriverWait(self.driver, 15)
        try:
            self.game_el = wait.until(EC.presence_of_element_located((By.TAG_NAME, 'ruffle-object')))
        except Exception:
            self.game_el = wait.until(EC.presence_of_element_located((By.TAG_NAME, 'ruffle-embed')))

        # Focus + start (space + click)
        ActionChains(self.driver).click(on_element=self.game_el).perform()
        sleep(0.3)
        self._tap_space()
        sleep(0.3)

        ActionChains(self.driver).click(on_element=self.game_el).perform()
        sleep(0.5)

        # --- Capture bbox via DOM rect ---
        rect = self.driver.execute_script("""
            const r = arguments[0].getBoundingClientRect();
            return {left: Math.floor(r.left), top: Math.floor(r.top),
                    width: Math.floor(r.width), height: Math.floor(r.height)};
        """, self.game_el)
        self.capture_bbox = {
            "top": int(rect["top"]),
            "left": int(rect["left"]),
            "width": int(rect["width"]),
            "height": int(rect["height"])
        }

        # progress proxy 상태
        self.last_progress = 0.0
        self.last_improve_time = time()
        self.episode_start_time = time()

    # -------- Low-level utilities --------
    def _tap_space(self):
        self.keyboard.press(Key.space); self.keyboard.release(Key.space)

    def _press_keys(self, keys):
        for k in keys: self.keyboard.press(k)

    def _release_keys(self, keys):
        for k in keys: self.keyboard.release(k)

    def _grab_gray(self) -> np.ndarray:
        """현재 게임 화면을 단일 채널(gray proxy)로 캡처 (R 채널 사용)"""
        with mss() as sct:
            shot = sct.grab(self.capture_bbox)
        arr = np.array(shot)[:, :, 2]  # BGRA -> R 채널
        return arr

    # -------- Reward/Done (proxy 버전) --------
    def _progress_proxy(self, gray: np.ndarray) -> float:
        """
        매우 단순한 진행도 프록시:
        화면 오른쪽 10%의 밝은 픽셀 수. (전진 시 배경/라인 변화로 값 변화 유도)
        """
        h, w = gray.shape
        roi = gray[:, int(w*0.90):]
        return float((roi > 200).sum())

    def _done_check(self) -> bool:
        # idle로 인한 종료
        if (time() - self.last_improve_time) > self.idle_done_sec:
            return True
        # 에피소드 타임아웃
        if (time() - self.episode_start_time) > self.step_timeout_sec:
            return True
        return False

    # -------- Gym-like API --------
    def reset(self) -> np.ndarray:
        self._tap_space()             # 리셋
        sleep(0.8)
        self.frames.clear()
        gray = self._grab_gray()
        obs = preprocess_frame(gray)
        for _ in range(self.stack):
            self.frames.append(obs)
        self.last_progress = self._progress_proxy(gray)
        self.last_improve_time = time()
        self.episode_start_time = time()
        return np.stack(list(self.frames), axis=0)  # (stack, 80, 80)

    def step(self, action: int):
        keys = ACTIONS.get(action, [])
        self._press_keys(keys)
        sleep(self.action_hold_sec)
        self._release_keys(keys)

        gray = self._grab_gray()
        obs = preprocess_frame(gray)
        self.frames.append(obs)
        stacked = np.stack(list(self.frames), axis=0)

        prog = self._progress_proxy(gray)
        delta = prog - self.last_progress
        reward = float(delta)
        if delta > 1e-6:
            self.last_improve_time = time()
        self.last_progress = prog

        done = self._done_check()
        info: Dict = {}
        return stacked, reward, done, info

# ----------------- quick test -----------------
if __name__ == '__main__':
    env = QWOPEnv()
    obs = env.reset()
    print("reset obs:", obs.shape)

    total_r = 0.0
    for t in range(50):
        a = np.random.randint(0, len(ACTIONS))
        obs, r, done, info = env.step(a)
        total_r += r
        print(f"t={t:02d} a={a} r={r:.3f} done={done}")
        if done:
            break
    print("total_r:", total_r)

    # 스택 1장만 저장해보기
    # Image.fromarray((obs[-1]*255).astype(np.uint8)).save("qwop_obs_last.png")
