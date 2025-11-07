# QWOP Original Environment (Selenium + OCR) — CDP 키입력(동시 누름 지원)
# gym과 분리된 native env

from typing import Union, Tuple, Dict, Any, List
import os, re, random
import numpy as np
import cv2
import pytesseract
from PIL import Image
from time import sleep, time
from mss import mss

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains

from collections import deque


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


class QWOPEnv:
    # 상단 거리 텍스트 ROI (OCR용) - 비율
    ROI_Y0, ROI_Y1 = 0.07, 0.20
    ROI_X0, ROI_X1 = 0.30, 0.70

    def __init__(self, debug_ocr: bool = False, frame_stack: int = 1, background_safe: bool = True):
        self.debug_ocr = debug_ocr
        self.frame_stack = frame_stack
        self.have_valid_dist = False
        self.background_safe = background_safe
        self.frame_buffer = deque(maxlen=frame_stack)

        # 브라우저
        opts = webdriver.ChromeOptions()
        # opts.add_argument("--headless=new")
        opts.add_argument("--window-size=1000,1000")
        # opts.add_argument("--window-position=3000,100")
        opts.add_argument("--disable-background-timer-throttling")
        opts.add_argument("--disable-renderer-backgrounding")
        opts.add_argument("--disable-backgrounding-occluded-windows")

        self.driver = webdriver.Chrome(options=opts)
        self.driver.get("http://0.0.0.0:8000/")

        wait = WebDriverWait(self.driver, 10)
        game_obj = wait.until(EC.element_to_be_clickable((By.TAG_NAME, "ruffle-object")))
        self.game_elem = game_obj  # 포커스 대상 저장
        ActionChains(self.driver).click(on_element=self.game_elem).perform()
        sleep(0.5)

        # CDP 키맵
        self._KEYMAP: Dict[str, Dict[str, Any]] = {
            'q': dict(key='q', code='KeyQ', keyCode=81),
            'w': dict(key='w', code='KeyW', keyCode=87),
            'o': dict(key='o', code='KeyO', keyCode=79),
            'p': dict(key='p', code='KeyP', keyCode=80),
            ' ': dict(key=' ', code='Space', keyCode=32),
        }

        # 최초 시작(스페이스) - CDP로 전송
        ActionChains(self.driver).click(on_element=self.game_elem).perform()
        self._cdp_key_down(' '); self._cdp_key_up(' ')
        sleep(2)
        ActionChains(self.driver).click(on_element=self.game_elem).perform()
        sleep(0.5)

        # mss 캡처 영역
        # Selenium get_attribute("width"/"height")가 문자열일 수 있으니 int 변환
        width = int(self.game_elem.get_attribute("width") or 800)
        height = int(self.game_elem.get_attribute("height") or 600)
        loc = self.game_elem.location
        # 필요시 offset 조정
        self.game_obj_location = {
            "top": int(loc['y']) + 200,
            "left": int(loc['x']) + 100,
            "width": width,
            "height": height,
        }

        # mss 재사용
        self.sct = mss()

        # 상태 변수
        self.prev_dist = np.nan
        self._dist_before = np.nan
        self.last_improve_time = time()
        self.episode_start_time = time()
        self.idle_done_sec = 10.0
        self.step_timeout_sec = 12.0
        self.baseline_bright = 0.10
        self.nan_streak = 0
        self.nan_done_streak = 50000000000  # 연속 5회 NaN이면 done (의도적으로 매우 큼)

        # 주기/속도 파라미터
        self.key_hold = 0.1      # 키 유지시간(초) 0.08~0.15 권장
        self.dt = 1.0/30.0       # 스텝 주기(30Hz)
        self.ocr_stride = 4      # 매 4스텝마다 한 번만 OCR
        self.step_i = 0

        # 관측 사이즈 축소(scale)
        self.obs_scale = 0.25    # 관측 프레임 축소 비율(0.25 = 1/4)

        self.save_dir = os.getcwd()

        self.fall_penalty = 3.0

    # ====== Public API ======
    def reset(self) -> np.ndarray:
        self._restart_game()
        self.frame_buffer.clear()
        frames = self._capture_frames_raw(self.frame_stack, force_ocr=True)
        for frame in frames:
            self.frame_buffer.append(frame)

        obs = np.stack(list(self.frame_buffer), axis=0)
        self.step_i = 0
        self.nan_streak = 0
        return obs

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        t0 = time()
        keys = ACTIONS.get(action, [])

        if keys:
            self._press_combo(keys)  # 동시 누름: 모두 down → 대기 → 모두 up
        else:
            sleep(self.key_hold)     # no-op

        do_ocr = (self.step_i % self.ocr_stride == 0)

        # 새 프레임만 캡처
        new_frame_list = self._capture_frames_raw(1, force_ocr=do_ocr)
        new_frame = new_frame_list[0]

        # 버퍼 업데이트
        self.frame_buffer.append(new_frame)

        # 관측 상태
        obs = np.stack(list(self.frame_buffer), axis=0)
        gray = new_frame

        curr_dist = self.prev_dist  # OCR이 prev_dist 갱신
        if np.isnan(self._dist_before) or np.isnan(curr_dist):
            reward = 0.0
        else:
            reward = float(curr_dist - self._dist_before)
        self._dist_before = curr_dist

        # done 판정
        done = self._nan_done(curr_dist) or self._done_check(gray)
        if done:
            reward -= self.fall_penalty

        info = {"distance": float(curr_dist) if not np.isnan(curr_dist) else float("nan")}
        self.step_i += 1

        # 스텝 주기 맞추기
        remain = self.dt - (time() - t0)
        if remain > 0:
            sleep(remain)

        return obs, reward, done, info

    # ====== CDP Key Injection ======
    def _cdp_key_down(self, ch: str):
        k = self._KEYMAP[ch]
        self.driver.execute_cdp_cmd("Input.dispatchKeyEvent", {
            "type": "keyDown",
            "key": k["key"],
            "code": k["code"],
            "windowsVirtualKeyCode": k["keyCode"],
            "nativeVirtualKeyCode": k["keyCode"],
            "text": k["key"] if len(k["key"]) == 1 else ""
        })

    def _cdp_key_up(self, ch: str):
        k = self._KEYMAP[ch]
        self.driver.execute_cdp_cmd("Input.dispatchKeyEvent", {
            "type": "keyUp",
            "key": k["key"],
            "code": k["code"],
            "windowsVirtualKeyCode": k["keyCode"],
            "nativeVirtualKeyCode": k["keyCode"],
            "text": k["key"] if len(k["key"]) == 1 else ""
        })

    def _press_combo(self, keys: List[str]):
        # 포커스 보장
        ActionChains(self.driver).click(on_element=self.game_elem).perform()
        # 동시 누름
        for k in keys:
            self._cdp_key_down(k)
        sleep(self.key_hold)
        for k in reversed(keys):
            self._cdp_key_up(k)

    # ====== Low-level ======
    def _restart_game(self):
        print("\n================= EPISODE END / RESTART =================\n")
        t0 = time()
        deadline = 2.0
        while time() - t0 < deadline:
            ActionChains(self.driver).click(on_element=self.game_elem).perform()
            sleep(0.5)
            self._cdp_key_down(' '); self._cdp_key_up(' ')
            sleep(0.3)
            # 시작 감지(팝업 사라짐 또는 거리 읽힘)
            raw = self.sct.grab(self.game_obj_location)
            arr_full = np.asarray(raw)[:, :, 0]
            if not self._is_restart_popup(arr_full):
                d = self._try_ocr_once(arr_full)
                if not np.isnan(d):
                    break

        self.prev_dist = np.nan
        self._dist_before = np.nan
        self.last_improve_time = time()
        self.episode_start_time = time()
        self.nan_streak = 0

    def _nan_done(self, dist: float) -> bool:
        if np.isnan(dist):
            self.nan_streak += 1
        else:
            self.nan_streak = 0
        return self.nan_streak >= self.nan_done_streak

    def _capture_frames_raw(self, n: int, force_ocr: bool=False) -> np.ndarray:
        """관측 프레임은 축소본으로 리턴. OCR은 항상 원본(arr_full)에서만 수행."""
        frames = []
        for _ in range(n):
            raw = self.sct.grab(self.game_obj_location)       # BGRA
            arr_full = np.asarray(raw)[:, :, 0]               # 원본 1채널(B)

            # 관측은 축소본으로
            arr_obs = cv2.resize(
                arr_full, (0, 0),
                fx=self.obs_scale, fy=self.obs_scale,
                interpolation=cv2.INTER_AREA
            )
            frames.append(arr_obs)

            # OCR → prev_dist 갱신 및 improve 타이머 갱신 (필요시에만)
            if force_ocr:
                dist = self._ocr_distance(arr_full)
                if self.debug_ocr:
                    print(f"[OCR] distance: {dist} metres")

        return np.stack(frames, axis=0)

    # ====== OCR / Popup / Done ======
    def _try_ocr_once(self, gray_full: np.ndarray) -> float:
        """재시작 확인용 빠른 한 번 읽기(상태 갱신은 하지 않음)."""
        h, w = gray_full.shape[:2]
        y0, y1 = int(h*0.07), int(h*0.20)
        x0, x1 = int(w*0.30), int(w*0.70)
        roi = gray_full[y0:y1, x0:x1]
        up = cv2.resize(roi, None, fx=3, fy=3, interpolation=cv2.INTER_LANCZOS4)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(up)
        _, bw = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cfg = "--oem 3 --psm 6 -l eng -c tessedit_char_blacklist=ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        data = pytesseract.image_to_data(bw, config=cfg, output_type=pytesseract.Output.DICT)
        best_val = None; best_conf = -1.0
        for i in range(len(data["text"])):
            t = (data["text"][i] or "").lower().strip()
            if t in ("metre", "metres"):
                line_i = data["line_num"][i]
                for j in range(i-1, -1, -1):
                    if data["line_num"][j] != line_i: break
                    tj = (data["text"][j] or "").lower().strip()
                    if not tj: continue
                    tj = tj.replace("o.", "0.").replace(",", ".")
                    m = re.fullmatch(r"-?\d+\.\d+", tj)
                    if m:
                        val = float(m.group(0))
                        conf = float(data["conf"][j]) if data["conf"][j] != "-1" else 60.0
                        if conf > best_conf:
                            best_conf, best_val = conf, val
                        break
        return float("nan") if best_val is None else float(best_val)

    def _ocr_distance(self, gray_full: np.ndarray) -> float:
        h, w = gray_full.shape[:2]
        y0, y1 = int(h*0.07), int(h*0.20)
        x0, x1 = int(w*0.30), int(w*0.70)
        roi = gray_full[y0:y1, x0:x1]

        up = cv2.resize(roi, None, fx=3, fy=3, interpolation=cv2.INTER_LANCZOS4)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(up)
        _, bw = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        cfg = "--oem 3 --psm 6 -l eng -c tessedit_char_blacklist=ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        data = pytesseract.image_to_data(bw, config=cfg, output_type=pytesseract.Output.DICT)

        best_val, best_conf = None, -1.0
        for i in range(len(data["text"])):
            t = (data["text"][i] or "").lower().strip()
            if t in ("metre", "metres"):
                line_i = data["line_num"][i]
                for j in range(i-1, -1, -1):
                    if data["line_num"][j] != line_i: break
                    tj = (data["text"][j] or "").lower().strip()
                    if not tj: continue
                    tj = tj.replace("o.", "0.").replace(",", ".")
                    m = re.fullmatch(r"-?\d+\.\d+", tj)  # 음수 허용
                    if m:
                        h_box = data["height"][j]
                        if h_box < 12: continue
                        conf = float(data["conf"][j]) if data["conf"][j] != "-1" else 60.0
                        val = float(m.group(0))
                        if conf > best_conf:
                            best_conf, best_val = conf, val
                        break

        if best_val is None:
            return float("nan")

        prev = self.prev_dist
        if not np.isnan(prev):
            # 개선이면 타이머 갱신
            if best_val > prev + 1e-4:
                self.last_improve_time = time()
            # 한 프레임 급점프는 노이즈로 처리
            if abs(best_val - prev) > 1.0:
                return float("nan")

        self.prev_dist = best_val
        return best_val

    def _roi_popup(self, gray: np.ndarray) -> np.ndarray:
        h, w = gray.shape[:2]
        y0, y1 = int(h*0.10), int(h*0.90)
        x0, x1 = int(w*0.10), int(w*0.90)
        return gray[y0:y1, x0:x1]

    def _is_restart_popup(self, gray: np.ndarray) -> bool:
        roi = self._roi_popup(gray)
        if roi.size == 0: return False
        bright_ratio = (roi > 210).mean()
        rh, rw = roi.shape
        mid_band = roi[int(rh*0.42):int(rh*0.60), int(rw*0.15):int(rw*0.85)]
        dark_mid_ratio = (mid_band < 80).mean() if mid_band.size else 0.0
        b = 6
        if rh > 2*b and rw > 2*b:
            border = np.concatenate([roi[:, :b].ravel(), roi[:, -b:].ravel(),
                                     roi[:b, :].ravel(), roi[-b:, :].ravel()])
            edge_contrast = abs(border.mean() - roi[b:-b, b:-b].mean())
        else:
            edge_contrast = 0.0
        base_bright = getattr(self, "baseline_bright", 0.0)
        cond_bright = (bright_ratio > base_bright + 0.12) or (bright_ratio > 0.22)
        cond_dark = dark_mid_ratio > 0.06
        cond_edge = edge_contrast > 20
        return cond_bright and (cond_dark or cond_edge)

    def _done_check(self, gray: np.ndarray) -> bool:
        if self._is_restart_popup(gray):
            return True
        return False


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
        print("stop")
