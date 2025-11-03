# QWOPEnv_gymlike.py
from typing import Union, Tuple, Dict
from collections import deque
import numpy as np
import cv2
from time import sleep, time
from mss import mss
from PIL import Image
import os
import re
import pytesseract

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from pynput.keyboard import Controller, Key


ACTIONS = {
    0: [],
    1: ['q'],
    2: ['w'],
    3: ['o'],
    4: ['p'],
    5: ['q','w'],
    6: ['q','o'],
    7: ['q','p'],
    8: ['w','o'],
    9: ['w','p'],
    10: ['o','p'],
    11: ['q','w','o'],
    12: ['q','w','p'],
    13: ['q','o','p'],
    14: ['w','o','p'],
    15: ['q','w','o','p'],
}


def preprocess_frame(gray: np.ndarray, out_hw: Tuple[int,int]=(80,80)) -> np.ndarray:
    img = cv2.resize(gray, out_hw, interpolation=cv2.INTER_AREA)
    return (img.astype(np.float32) / 255.0)


class QWOPEnv:
    """Gym 스타일 step/reset 지원하는 QWOP 환경 (Ruffle + SWF 화면 캡처 기반)"""
    def __init__(self,
                 url: str = 'http://0.0.0.0:8000/',
                 stack: int = 4,
                 action_hold_sec: float = 0.12,
                 idle_done_sec: float = 3.0,
                 step_timeout_sec: float = 30.0,
                 debug_ocr: bool = False):
        self.stack = stack
        self.frames = deque(maxlen=stack)
        self.action_hold_sec = action_hold_sec
        self.idle_done_sec = idle_done_sec
        self.step_timeout_sec = step_timeout_sec
        self.prev_dist = np.nan
        self.debug_ocr = debug_ocr
        if self.debug_ocr:
            os.makedirs("ocr_dbg", exist_ok=True)

        # --- Browser ---
        options = webdriver.ChromeOptions()
        options.add_argument("--window-size=1000,1000")
        self.driver = webdriver.Chrome(options=options)
        try:
            self.driver.set_window_position(0, 0)
        except Exception:
            pass
        self.driver.get(url)

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

        # --- Capture bbox via DOM rect (DPR 보정 포함) ---
        rect, dpr = self.driver.execute_script("""
          const r = arguments[0].getBoundingClientRect();
          const dpr = window.devicePixelRatio || 1;
          return [
            {left: r.left, top: r.top, width: r.width, height: r.height},
            dpr
          ];
        """, self.game_el)

        L = int(round(rect["left"] * dpr))
        T = int(round(rect["top"] * dpr))
        W = int(round(rect["width"] * dpr))
        H = int(round(rect["height"] * dpr))
        self.capture_bbox = {"top": T, "left": L, "width": W, "height": H}

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
        with mss() as sct:
            shot = sct.grab(self.capture_bbox)
        bgra = np.array(shot)  # (H,W,4)
        return cv2.cvtColor(bgra, cv2.COLOR_BGRA2GRAY)

    # -------- Reward/Done (proxy 버전) --------
    def _progress_proxy(self, gray: np.ndarray) -> float:
        h, w = gray.shape
        roi = gray[:, int(w*0.90):]
        return float((roi > 200).sum())

    # === 종료 팝업 감지 ===
    def _is_restart_popup(self, gray: np.ndarray) -> bool:
        roi = self._roi(gray)
        if roi.size == 0:
            return False
        bright_ratio = (roi > 210).mean()
        rh, rw = roi.shape
        mid_band = roi[int(rh * 0.42):int(rh * 0.60), int(rw * 0.15):int(rw * 0.85)]
        dark_mid_ratio = (mid_band < 80).mean() if mid_band.size else 0.0
        b = 6
        if rh > 2 * b and rw > 2 * b:
            left = roi[:, :b].ravel(); right = roi[:, -b:].ravel()
            top = roi[:b, :].ravel();  bottom = roi[-b:, :].ravel()
            border_mean = np.concatenate([left, right, top, bottom]).mean()
            body_mean = roi[b:-b, b:-b].mean()
            edge_contrast = abs(border_mean - body_mean)
        else:
            edge_contrast = 0.0
        base_bright = getattr(self, "baseline_bright", 0.0)
        cond_bright = (bright_ratio > base_bright + 0.12) or (bright_ratio > 0.22)
        cond_dark = dark_mid_ratio > 0.06
        cond_edge = edge_contrast > 20
        return cond_bright and (cond_dark or cond_edge)

    def _done_check(self, gray: np.ndarray) -> bool:
        if (time() - self.last_improve_time) > self.idle_done_sec:
            return True
        if (time() - self.episode_start_time) > self.step_timeout_sec:
            return True
        if self._is_restart_popup(gray):
            return True
        return False

    # === OCR: 거리 텍스트 ROI ===
    def _distance_roi(self, gray: np.ndarray) -> Tuple[np.ndarray, Tuple[int,int,int,int]]:
        h, w = gray.shape
        y0, y1 = int(h * 0.07), int(h * 0.20)
        x0, x1 = int(w * 0.30), int(w * 0.70)
        return gray[y0:y1, x0:x1], (x0, y0, x1, y1)

    def _ocr_distance(self, arr: np.ndarray, debug_once: bool = True) -> float:
        """
        스탠드얼론에서 잘 되던 파이프라인 그대로:
        - GRAY 입력 사용
        - ROI: 0.07~0.20, 0.30~0.70
        - 업스케일+CLAHE+OTSU
        - 같은 라인의 'metres' 왼쪽 소수만 추출
        """
        h, w = arr.shape[:2]
        y0, y1 = int(h * 0.07), int(h * 0.20)
        x0, x1 = int(w * 0.30), int(w * 0.70)
        roi = arr[y0:y1, x0:x1]  # GRAY

        up = cv2.resize(roi, None, fx=3, fy=3, interpolation=cv2.INTER_LANCZOS4)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(up)
        _, bw = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        if self.debug_ocr and debug_once:
            cv2.imwrite("ocr_dbg/roi_gray.png", roi)
            cv2.imwrite("ocr_dbg/roi_up.png", up)
            cv2.imwrite("ocr_dbg/roi_bw.png", bw)

        cfg = "--oem 3 --psm 6 -l eng -c tessedit_char_blacklist=ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        data = pytesseract.image_to_data(bw, config=cfg, output_type=pytesseract.Output.DICT)

        best_val, best_conf = None, -1.0
        n = len(data["text"])
        for i in range(n):
            t = (data["text"][i] or "").lower().strip()
            if t in ("metre", "metres"):
                line_i = data["line_num"][i]
                for j in range(i - 1, -1, -1):
                    if data["line_num"][j] != line_i:
                        break
                    tj = (data["text"][j] or "").lower().strip()
                    if not tj:
                        continue
                    tj = tj.replace("o.", "0.").replace(",", ".")
                    m = re.fullmatch(r"\d+\.\d+", tj)
                    if m:
                        h_box = data["height"][j]
                        if h_box < 12:
                            continue
                        conf = float(data["conf"][j]) if data["conf"][j] != '-1' else 60.0
                        val = float(m.group(0))
                        if conf > best_conf:
                            best_conf, best_val = conf, val
                        break

        if best_val is None:
            return float("nan")

        prev = getattr(self, "prev_dist", np.nan)
        if not np.isnan(prev) and abs(best_val - prev) > 1.0:
            return float("nan")
        self.prev_dist = best_val
        return best_val

    # 컬러 캡처가 필요하면 사용
    def _grab_color(self) -> np.ndarray:
        with mss() as sct:
            shot = sct.grab(self.capture_bbox)
        return np.array(shot)  # BGRA

    # === 중앙 팝업 ROI ===
    def _roi(self, gray: np.ndarray):
        h, w = gray.shape
        y0, y1 = int(h * 0.45), int(h * 0.75)
        x0, x1 = int(w * 0.20), int(w * 0.80)
        return gray[y0:y1, x0:x1]

    def reset(self) -> np.ndarray:
        self._tap_space()
        sleep(0.8)
        self.frames.clear()
        gray = self._grab_gray()
        obs = preprocess_frame(gray)
        for _ in range(self.stack):
            self.frames.append(obs)
        self.last_progress = self._progress_proxy(gray)
        self.last_improve_time = time()
        self.episode_start_time = time()
        roi = self._roi(gray)
        self.baseline_mean = float(roi.mean())
        self.baseline_bright = float((roi > 210).mean())

        self.prev_dist = np.nan

        # 디버그로 시작 ROI 확인하고 싶으면 저장
        if self.debug_ocr:
            roi_top, _ = self._distance_roi(gray)
            cv2.imwrite("ocr_dbg/reset_distance_roi.png", roi_top)

        return np.stack(list(self.frames), axis=0)

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

        dist = self._ocr_distance(gray, debug_once=False)

        done = self._done_check(gray)
        info = {"distance": dist}
        return stacked, reward, done, info


# ----------------- quick test -----------------
if __name__ == '__main__':
    env = QWOPEnv(debug_ocr=False)
    try:
        ep = 0
        while True:
            obs = env.reset()
            print(f"[episode {ep}] reset obs:", obs.shape)

            total_r = 0.0
            t = 0
            while True:
                a = np.random.randint(0, len(ACTIONS))
                obs, r, done, info = env.step(a)
                total_r += r
                dist = info.get("distance", float("nan"))
                dist_s = "?.??" if np.isnan(dist) else f"{dist:.2f}"
                print(f"ep={ep:03d} t={t:04d} a={a:02d} r={r:.3f} dist={dist_s}m done={done}")
                t += 1

                if done:
                    print(f"[episode {ep}] done. total_r={total_r:.3f} steps={t}")
                    ep += 1
                    break

    except KeyboardInterrupt:
        print("종료(Ctrl+C). 브라우저는 수동으로 닫아도 됩니다.")
