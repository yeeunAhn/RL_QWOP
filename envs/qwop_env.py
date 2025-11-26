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
import re


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
    #5: ['q', 'w'],
    6: ['q', 'o'],
    7: ['q', 'p'],
    8: ['w', 'o'],
    9: ['w', 'p'],
    #10: ['o', 'p'],
    # 11: ['q','w','o'],
    # 12: ['q','w','p'],
    # 13: ['q','o','p'],
    # 14: ['w','o','p'],
    # 15: ['q','w','o','p'],
}


class QWOPEnv:
    # 상단 거리 텍스트 ROI (OCR용) - 비율
    ROI_Y0, ROI_Y1 = 0.07, 0.20
    ROI_X0, ROI_X1 = 0.30, 0.70

    def __init__(self, debug_ocr: bool = False, frame_stack: int = 1, background_safe: bool = True,
                 debug_posture: bool = False):
        self.debug_ocr = debug_ocr
        self.debug_posture = debug_posture
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
            'r': dict(key='r', code='KeyR', keyCode=82),
        }

        # 최초 시작(스페이스) - CDP로 전송
        ActionChains(self.driver).click(on_element=self.game_elem).perform()
        self._cdp_key_down(' ');
        self._cdp_key_up(' ')
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
        self.prev_dist = np.nan
        self._dist_before = np.nan
        self.last_improve_time = time()
        self.episode_start_time = time()
        self.idle_done_sec = 15.0
        self.step_timeout_sec = 100.0
        self.baseline_bright = 0.10
        self.nan_streak = 0
        self.nan_done_streak = 50000000000  # 연속 5회 NaN이면 done (의도적으로 매우 큼)

        # 주기/속도 파라미터
        self.key_hold = 0.08  # 키 유지시간(초) 0.08~0.15 권장
        self.dt = 1.0 / 30.0  # 스텝 주기(30Hz)
        self.ocr_stride = 6  # 매 6스텝마다 한 번만 OCR
        self.step_i = 0

        # 관측 사이즈 축소(scale)
        self.obs_scale = 0.15 # 관측 프레임 축소 비율(0.25 = 1/4)

        self.save_dir = os.getcwd()

        self.fall_penalty = 1.0  # 넘어질때

        # 자세 보상(Postural Reward) 관련 하이퍼파라미터
        self.posture_reward_scale = 0.1
        self.posture_roi_y = (0.20, 0.65)  # Y (20% ~ 65%) - 트랙 라인 피하기
        self.posture_roi_x = (0.25, 0.75)  # X (25% ~ 75%) - 캐릭터 중앙
        self.singlet_threshold = 210  # 흰색 싱레트 밝기 임계값 (튜닝 필요)
        self.ground_y_ratio = 0.68  # '땅'으로 간주할 Y 비율 (68%)

        # 디버그 창 초기화
        if self.debug_posture:
            self.posture_debug_window = "QWOP Posture Debug"
            self.mask_debug_window = "White Mask Debug"
            cv2.namedWindow(self.posture_debug_window, cv2.WINDOW_NORMAL)
            cv2.namedWindow(self.mask_debug_window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.posture_debug_window, 400, 300)
            cv2.resizeWindow(self.mask_debug_window, 200, 200)

    # ====== Public API ======
    def reset(self) -> np.ndarray:
        self._restart_game()
        self.frame_buffer.clear()

        frames, _ = self._capture_frames_raw(self.frame_stack, force_ocr=True)

        for frame in frames:
            self.frame_buffer.append(frame)

        obs = np.stack(list(self.frame_buffer), axis=0)
        self.step_i = 0
        self.nan_streak = 0

        # ⭐️ [추가] 이번 에피소드의 최고 도달 거리 (0부터 시작)
        self.max_dist_episode = 0.0

        return obs

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        t0 = time()
        keys = ACTIONS.get(action, [])

        if keys:
            self._press_combo(keys)
        else:
            sleep(self.key_hold)

        do_ocr = (self.step_i % self.ocr_stride == 0)

        new_frame_stack, last_full_frame = self._capture_frames_raw(1, force_ocr=do_ocr)
        new_frame = new_frame_stack[0]

        self.frame_buffer.append(new_frame)
        obs = np.stack(list(self.frame_buffer), axis=0)

        # ==========================================================
        # 💡 [업그레이드된 보상 로직]
        # ==========================================================

        # 1. 자세 보상 계산
        posture_reward = self._calculate_posture_reward(last_full_frame)
        curr_dist = self.prev_dist

        # 2. [수정] 생존 페널티 강화 (0.01 -> 0.05)
        #    이제 자세(0.1)만 잡고 안 움직이면 +0.05 밖에 못 법니다.
        #    빨리 앞으로 가서 거리 보상을 먹어야 이득이 커집니다.
        LIVING_PENALTY = 0.02
        reward = posture_reward - LIVING_PENALTY

        # 3. [추가] 최고 기록 갱신 보상 (Max Distance Reward)
        #    와리가리는 무시하고, "새로운 땅"을 밟을 때만 보상!
        STEP_REWARD_SCALE = 4.0  # (가중치)

        if not np.isnan(curr_dist):
            # 만약 이번 스텝의 거리가 '지금까지의 최고 기록'보다 크다면?
            if curr_dist > self.max_dist_episode:
                diff = curr_dist - self.max_dist_episode

                # 갱신한 만큼 보상 지급 (0.1m 갱신하면 +0.5점)
                reward += diff * STEP_REWARD_SCALE

                # 최고 기록 업데이트
                self.max_dist_episode = curr_dist

        # step() 안에서 curr_dist 계산 후, reward 계산 부분 바로 아래에 추가

        STEP_PROGRESS_SCALE = 3.0  # 새로 추가 (5보다 작게 시작 추천)

        dist_progress = 0.0
        if not np.isnan(curr_dist) and not np.isnan(self._dist_before):
            dist_progress = curr_dist - self._dist_before
            if dist_progress > 0:
                reward += dist_progress * STEP_PROGRESS_SCALE

        self._dist_before = curr_dist

        # ==========================================================

        # 1. done 판정
        done = self._nan_done(curr_dist) or self._done_check(last_full_frame)
        final_dist_for_info = curr_dist

        # 1. 경과 시간 계산
        time_since_improve = time() - self.last_improve_time
        time_since_start = time() - self.episode_start_time

        # 2. 동적 제한 시간 계산
        # 기본 20초 + (현재 거리 * 15초)
        # 예: 0m -> 20초 제한
        # 예: 2m -> 20 + 30 = 50초 제한
        # 예: 10m -> 20 + 150 = 170초 제한
        current_dist_val = curr_dist if not np.isnan(curr_dist) else 0.0
        dynamic_time_limit = 20.0 + (max(0, current_dist_val) * 15.0)

        # 3. Done 판정
        # (조건 A) 오랫동안 제자리걸음(idle) 이거나
        # (조건 B) 거리에 비해 시간이 너무 많이 지났으면(dynamic_limit) 종료
        if not done:
            if time_since_improve > self.idle_done_sec:
                print(f"[Env] Done: Idle for {time_since_improve:.1f}s (No progress)")
                done = True

            elif time_since_start > dynamic_time_limit:
                print(
                    f"[Env] Done: Time Over! ({time_since_start:.1f}s > Limit {dynamic_time_limit:.1f}s for {current_dist_val:.2f}m)")
                done = True

        if done:
            reward -= self.fall_penalty

            sleep(0.2)
            raw_last = self.sct.grab(self.game_obj_location)
            gray_last = np.asarray(raw_last)[:, :, 0]

            # 이 gray_last 기준으로 최종 점수 읽기
            final_score = self._try_ocr_once(gray_last)
            if not np.isnan(final_score):
                final_dist_for_info = final_score

            FINAL_DISTANCE_REWARD_SCALE = 10.0
            final_dist_reward = 0.0
            if not np.isnan(final_dist_for_info):
                final_dist_reward = final_dist_for_info * FINAL_DISTANCE_REWARD_SCALE

            reward += final_dist_reward
            print(f"[Env] Final dist reward: {final_dist_reward:.3f} (dist: {final_dist_for_info:.2f})")

            if self._is_restart_popup(gray_last):
                print("[Env] Restart popup detected at end of episode.")

        info = {
            "distance": float(final_dist_for_info) if not np.isnan(final_dist_for_info) else float("nan"),
            "posture_reward": posture_reward
        }
        self.step_i += 1

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

    def _restart_game(self):
        print("\n================= EPISODE END / RESTART =================\n")

        # 재시작 시도 최대 3회
        for try_count in range(3):
            try:
                # 1. 포커스 확보
                if hasattr(self, 'game_elem') and self.game_elem:
                    ActionChains(self.driver).click(on_element=self.game_elem).perform()
                else:
                    ActionChains(self.driver).click(self.driver.find_element(By.TAG_NAME, "body")).perform()

                sleep(0.2)

                # 2. 'R' 키 전송 (재시작)
                self._cdp_key_down('r');
                sleep(0.05);
                self._cdp_key_up('r')
                print(f"[Env] 'R' key pressed (Attempt {try_count + 1})")

                # 3. 잠시 대기 후 화면 확인
                sleep(1.0)

                # 4. [⭐️ 핵심] 점수 확인 ("좀비 감지")
                raw = self.sct.grab(self.game_obj_location)
                arr = np.asarray(raw)[:, :, 0]

                check_dist = self._try_ocr_once(arr)

                # ⭐️ [수정] 0.2 -> 0.5 로 변경 (0.3m 시작도 정상으로 인식하게)
                if np.isnan(check_dist) or check_dist < 0.5:
                    print("[Env] Restart Success (Score reset).")
                    break  # 루프 탈출
                else:
                    print(f"[Env] Restart Failed? Score is still {check_dist}m. Retrying...")

            except Exception as e:
                print(f"[Env] Error during soft restart: {e}")

            # 3번 다 실패했거나 에러나면 -> 강제 새로고침 (F5)
            if try_count == 2:
                print("[Env] 🚨 ZOMBIE DETECTED! Force Refreshing Page (F5)...")
                try:
                    self.driver.refresh()
                    sleep(3.0)

                    # 게임 요소 다시 찾기
                    wait = WebDriverWait(self.driver, 10)
                    self.game_elem = wait.until(EC.element_to_be_clickable((By.TAG_NAME, "ruffle-object")))
                    ActionChains(self.driver).click(on_element=self.game_elem).perform()
                    sleep(0.5)

                    # 스페이스바로 시작
                    self._cdp_key_down(' ');
                    self._cdp_key_up(' ')
                    sleep(1.0)
                except:
                    pass

        # 상태 변수 초기화
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

    def _capture_frames_raw(self, n: int, force_ocr: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        관측 프레임은 축소본 스택으로,
        마지막 원본 프레임은 별도로 리턴. (자세 보상 계산용)
        """
        frames = []
        last_arr_full = None
        for _ in range(n):
            raw = self.sct.grab(self.game_obj_location)  # BGRA
            arr_full = np.asarray(raw)[:, :, 0]  # 원본 1채널(B)
            last_arr_full = arr_full

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

        # 스택된 프레임과 마지막 원본 프레임을 반환
        return np.stack(frames, axis=0), last_arr_full

    # ====== OCR / Popup / Done ======
    def _try_ocr_once(self, gray_full: np.ndarray) -> float:
        """재시작 확인용 빠른 한 번 읽기(상태 갱신은 하지 않음)."""
        h, w = gray_full.shape[:2]
        y0, y1 = int(h * self.ROI_Y0), int(h * self.ROI_Y1)
        x0, x1 = int(w * self.ROI_X0), int(w * self.ROI_X1)
        roi = gray_full[y0:y1, x0:x1]
        up = cv2.resize(roi, None, fx=3, fy=3, interpolation=cv2.INTER_LANCZOS4)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(up)
        _, bw = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cfg = "--oem 3 --psm 6 -l eng -c tessedit_char_blacklist=ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        data = pytesseract.image_to_data(bw, config=cfg, output_type=pytesseract.Output.DICT)
        best_val = None;
        best_conf = -1.0
        for i in range(len(data["text"])):
            t = (data["text"][i] or "").lower().strip()
            if t in ("metre", "metres"):
                line_i = data["line_num"][i]
                for j in range(i - 1, -1, -1):
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
        y0, y1 = int(h * self.ROI_Y0), int(h * self.ROI_Y1)
        x0, x1 = int(w * self.ROI_X0), int(w * self.ROI_X1)
        roi = gray_full[y0:y1, x0:x1]

        up = cv2.resize(roi, None, fx=3, fy=3, interpolation=cv2.INTER_LANCZOS4)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(up)
        _, bw = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        cfg = "--oem 3 --psm 6 -l eng -c tessedit_char_blacklist=ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        data = pytesseract.image_to_data(bw, config=cfg, output_type=pytesseract.Output.DICT)

        best_val, best_conf = None, -1.0
        for i in range(len(data["text"])):
            t = (data["text"][i] or "").lower().strip()
            if t in ("metre", "metres"):
                line_i = data["line_num"][i]
                for j in range(i - 1, -1, -1):
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

    def _calculate_posture_reward(self, gray_full: np.ndarray) -> float:
        """
        '흰색 싱레트'의 높이를 기반으로 자세 보상을 계산합니다.
        보상은 0.0 (나쁨) ~ self.posture_reward_scale (좋음) 사이입니다.
        """
        if gray_full is None:
            return 0.0

        h, w = gray_full.shape

        # 1. 자세 ROI 정의 (트랙 라인 제외)
        roi_y0 = int(h * self.posture_roi_y[0])
        roi_y1 = int(h * self.posture_roi_y[1])
        roi_x0 = int(w * self.posture_roi_x[0])
        roi_x1 = int(w * self.posture_roi_x[1])

        torso_roi = gray_full[roi_y0:roi_y1, roi_x0:roi_x1]

        if torso_roi.size == 0:
            return 0.0

        # 2. 흰색 싱레트 마스크 생성
        white_mask = (torso_roi > self.singlet_threshold).astype(np.uint8)

        # 3. 흰색 픽셀 좌표 찾기
        white_pixels_y, _ = np.where(white_mask > 0)

        avg_singlet_y = 0.0
        norm_height = 0.0
        posture_reward = 0.0  # 정규화된 보상 (0~1)

        if white_pixels_y.size < 10:  # 감지된 픽셀이 너무 적으면 무시
            posture_reward = 0.0
        else:
            # 4. 싱레트의 평균 높이 계산 (노이즈에 강하도록 평균 사용)
            avg_singlet_y = np.mean(white_pixels_y) + roi_y0  # 원본 좌표계로 복원

            # 5. 보상 계산
            ground_y_level = int(h * self.ground_y_ratio)
            ideal_y_level = roi_y0
            norm_height = (ground_y_level - avg_singlet_y) / (ground_y_level - ideal_y_level + 1e-6)
            posture_reward = np.clip(norm_height, 0.0, 1.0)

        # 디버그 시각화 로직
        if self.debug_posture:
            debug_img = cv2.cvtColor(gray_full, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(debug_img, (roi_x0, roi_y0), (roi_x1, roi_y1), (0, 255, 0), 1)
            ground_y_level = int(h * self.ground_y_ratio)
            ideal_y_level = roi_y0
            cv2.line(debug_img, (0, ground_y_level), (w, ground_y_level), (0, 0, 255), 1)
            cv2.line(debug_img, (0, ideal_y_level), (w, ideal_y_level), (255, 0, 0), 1)
            if white_pixels_y.size > 10:
                cv2.circle(debug_img, (w // 2, int(avg_singlet_y)), 5, (0, 255, 255), -1)

            final_p_reward = posture_reward * self.posture_reward_scale

            cv2.putText(debug_img,
                        f"Posture R: {final_p_reward:.3f} (Raw: {posture_reward:.2f} * {self.posture_reward_scale:.2f})",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(debug_img, f"Norm_H: {norm_height:.3f}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            cv2.imshow(self.posture_debug_window, debug_img)
            cv2.imshow(self.mask_debug_window, white_mask * 255)
            cv2.waitKey(1)

        # 최종 보상 스케일 적용 (0~1 사이 값 * 스케일)
        return posture_reward * self.posture_reward_scale

    def _roi_popup(self, gray: np.ndarray) -> np.ndarray:
        h, w = gray.shape[:2]
        y0, y1 = int(h * 0.10), int(h * 0.90)
        x0, x1 = int(w * 0.10), int(w * 0.90)
        return gray[y0:y1, x0:x1]

    def _is_restart_popup(self, gray: np.ndarray) -> bool:
        roi = self._roi_popup(gray)
        if roi.size == 0: return False
        bright_ratio = (roi > 210).mean()
        rh, rw = roi.shape
        mid_band = roi[int(rh * 0.42):int(rh * 0.60), int(rw * 0.15):int(rw * 0.85)]
        dark_mid_ratio = (mid_band < 80).mean() if mid_band.size else 0.0
        b = 6
        if rh > 2 * b and rw > 2 * b:
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

    # 환경 종료 메서드
    def close(self):
        """환경을 종료하고 창을 닫습니다."""
        print("Cleaning up environment and closing windows...")
        if hasattr(self, 'driver'):
            self.driver.quit()
        cv2.destroyAllWindows()


# ----------------- quick test -----------------
if __name__ == '__main__':
    # debug_posture=True로 환경 생성 (디버그 창 활성화)
    env = QWOPEnv(debug_ocr=False, frame_stack=1, background_safe=True, debug_posture=False)
    try:
        ep = 0
        while True:
            obs = env.reset()
            print(f"[episode {ep}] reset obs:", obs.shape)

            total_r = 0.0
            t = 0
            while True:
                a = np.random.randint(0, len(ACTIONS))
                obs, reward, done, info = env.step(a)
                total_r += reward
                dist = info.get("distance", float("nan"))
                posture_r = info.get("posture_reward", 0.0)

                dist_s = "?.??" if np.isnan(dist) else f"{dist:.2f}"
                # 로그에 자세 보상 추가
                print(f"ep={ep:03d} t={t:04d} a={a:02d} r={reward:.3f} (p:{posture_r:.3f}) dist={dist_s}m done={done}")
                t += 1

                if done:
                    print(f"[episode {ep}] done. total_r={total_r:.3f} steps={t}")
                    ep += 1
                    break

    except KeyboardInterrupt:
        print("stop")
    finally:
        # 종료 시 반드시 close 호출
        env.close()