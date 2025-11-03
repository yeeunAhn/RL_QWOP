from typing import Union, Tuple, Dict, Any
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


class QWOPEnv:
    # 상단 거리 텍스트 ROI (OCR용) - 비율
    ROI_Y0, ROI_Y1 = 0.07, 0.20
    ROI_X0, ROI_X1 = 0.30, 0.70

    def __init__(self, debug_ocr: bool=False, frame_stack: int=1):
        self.debug_ocr = debug_ocr
        self.frame_stack = frame_stack

        # 브라우저
        opts = webdriver.ChromeOptions()
        # opts.add_argument("--headless=new")
        opts.add_argument("--window-size=1000,1000")
        # 백그라운드 스로틀 완화(선택)
        opts.add_argument("--disable-background-timer-throttling")
        opts.add_argument("--disable-renderer-backgrounding")
        opts.add_argument("--disable-backgrounding-occluded-windows")

        self.driver = webdriver.Chrome(options=opts)
        self.driver.get("http://0.0.0.0:8000/")

        self.keyboard = Controller()

        wait = WebDriverWait(self.driver, 10)
        game_obj = wait.until(EC.element_to_be_clickable((By.TAG_NAME, "ruffle-object")))
        ActionChains(self.driver).click(on_element=game_obj).perform()
        sleep(0.5)

        # 최초 시작(스페이스)
        self.keyboard.press(Key.space); self.keyboard.release(Key.space)
        sleep(2)
        ActionChains(self.driver).click().perform()
        sleep(0.5)

        # mss 캡처 영역
        self.game_obj_location = {
            "top": game_obj.location['y'] + 200,
            "left": game_obj.location['x'] + 100,
            "width": int(game_obj.get_attribute("width")),
            "height": int(game_obj.get_attribute("height")),
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
        self.nan_done_streak = 5  # 연속 5회 NaN이면 done

        # 주기/속도 파라미터
        self.key_hold = 0.12     # 키 눌림 유지시간(초) 0.08~0.15 권장
        self.dt = 1.0/30.0       # 스텝 주기(30Hz)
        self.ocr_stride = 4      # 매 4스텝마다 한 번만 OCR
        self.step_i = 0

        # 관측 사이즈 축소(scale)
        self.obs_scale = 0.25    # 관측 프레임 축소 비율(0.25 = 1/4)

        self.save_dir = os.getcwd()

    # ====== Public API ======
    def reset(self) -> np.ndarray:
        self._restart_game()  # 구분선 출력 포함
        # 리셋 직후 초기 거리 한 번은 반드시 읽어둔다(강제 OCR)
        obs = self._capture_frames(self.frame_stack, force_ocr=True)
        self.step_i = 0
        self.nan_streak = 0
        return obs

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        t0 = time()
        keys = ACTIONS.get(action, [])
        for k in keys: self._press_key(k)
        sleep(self.key_hold)
        for k in keys: self._release_key(k)

        # OCR은 stride에 맞춰서만 수행
        do_ocr = (self.step_i % self.ocr_stride == 0)
        obs = self._capture_frames(1, force_ocr=do_ocr)
        gray = obs[0]

        curr_dist = self.prev_dist  # _capture_frames 내부 OCR이 prev_dist 갱신함
        # 보상: Δdistance (둘 중 하나라도 NaN이면 0)
        if np.isnan(self._dist_before) or np.isnan(curr_dist):
            reward = 0.0
        else:
            reward = float(curr_dist - self._dist_before)
        self._dist_before = curr_dist

        # done 판정
        done = self._nan_done(curr_dist) or self._done_check(gray)

        info = {"distance": float(curr_dist) if not np.isnan(curr_dist) else float("nan")}
        self.step_i += 1

        # 스텝 주기 맞추기(과도 루프 방지)
        remain = self.dt - (time() - t0)
        if remain > 0:
            sleep(remain)

        return obs, reward, done, info

    # ====== Low-level ======
    def _press_key(self, key: Union[str, Key]): self.keyboard.press(key)
    def _release_key(self, key: Union[str, Key]): self.keyboard.release(key)

    def _restart_game(self):
        print("\n================= EPISODE END / RESTART =================\n")
        # 포커스 보장 + 스페이스 재시도 루프(짧게)
        t0 = time()
        deadline = 2.0
        while time() - t0 < deadline:
            ActionChains(self.driver).click().perform()
            sleep(0.5)
            self.keyboard.press(Key.space); self.keyboard.release(Key.space)
            sleep(0.3)
            # 시작 감지(팝업 사라짐 또는 거리 읽힘)
            raw = self.sct.grab(self.game_obj_location)
            arr_full = np.asarray(raw)[:, :, 0]
            if not self._is_restart_popup(arr_full):
                # 한 번 읽혀보면 확실
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

    def _capture_frames(self, n: int, force_ocr: bool=False) -> np.ndarray:
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
                # 로그 줄이고 싶으면 아래 조건 달아:
                # if (t % 30)==0 or done:
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
