from typing import Union
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from pynput.keyboard import Controller, Key
import numpy as np
from PIL import Image
from time import sleep
from mss import mss

import re
import cv2, re, pytesseract, os




class QWOPEnv:
    def __init__(self):
        # Open browser
        options = webdriver.ChromeOptions()
        # options.add_argument('--headless')
        options.add_argument("--window-size=1000,1000")

        self.driver = webdriver.Chrome(options=options)
        self.driver.get(f'http://0.0.0.0:8000/')

        # Setup Keyboard
        self.keyboard = Controller()

        # Wait for game to load
        wait = WebDriverWait(self.driver, 10)
        game_obj = wait.until(
            EC.element_to_be_clickable((By.TAG_NAME, 'ruffle-object'))
        )

        # 🔹 click_and_hold 제거 → 단순 클릭만 수행
        actions = ActionChains(self.driver)
        actions.click(on_element=game_obj).perform()
        sleep(0.5)

        # 🔹 QWOP 시작: 스페이스로 실행
        self.keyboard.press(Key.space)
        self.keyboard.release(Key.space)
        sleep(2)
        actions = ActionChains(self.driver)
        actions.click().perform()
        sleep(0.5)

        # Get game object location and size
        self.game_obj_location = {
            "top": game_obj.location['y']+200,
            "left": game_obj.location['x']+100,
            "width": int(game_obj.get_attribute("width")),
            "height": int(game_obj.get_attribute("height"))
        }

    def _press_key(self, key: Union[str, Key]):
        self.keyboard.press(key)

    def _release_key(self, key: Union[str, Key]):
        self.keyboard.release(key)



        # 상단 숫자 ROI 비율 (필요하면 조절)

    ROI_Y0, ROI_Y1 = 0.07, 0.20  # 세로 7% ~ 20%
    ROI_X0, ROI_X1 = 0.30, 0.70  # 가로 30% ~ 70%

    def get_state(self, n: int) -> np.ndarray:
        np_list = []
        for i in range(n):
            with mss() as sct:
                screenshot = sct.grab(self.game_obj_location)

            # BGRA 중 B 채널(0)만 사용 → 단일 채널
            array_2d = np.array(screenshot)[:, :, 0]
            np_list.append(array_2d)

            # ---- 상단 숫자 ROI 잘라 별도 저장 ----
            h, w = array_2d.shape
            y0 = int(h * self.ROI_Y0);
            y1 = int(h * self.ROI_Y1)
            x0 = int(w * self.ROI_X0);
            x1 = int(w * self.ROI_X1)
            roi = array_2d[y0:y1, x0:x1]

            # 원본 프레임과 ROI 저장
            Image.fromarray(roi).save(f"distance_roi_{i}.png")

            # OCR distance 읽기
            dist = self._ocr_distance(array_2d)
            print(f"distance: {dist} metres")


        return np.stack(np_list, axis=0)

    import cv2, numpy as np, pytesseract, re, os

    # 상단에: import re, cv2, numpy as np, pytesseract
    # 클래스 __init__에: self.prev_dist = np.nan  # 이전 값 저장(이상치 필터용)

    def _ocr_distance(self, arr, debug_once=True) -> float:
        # --- ROI 추출 (지금 쓰는 비율 그대로) ---
        h, w = arr.shape[:2]
        y0, y1 = int(h * 0.07), int(h * 0.20)
        x0, x1 = int(w * 0.30), int(w * 0.70)
        roi = arr[y0:y1, x0:x1]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGRA2GRAY) if roi.ndim == 3 else roi

        # 전처리(업스케일+대비+이진화)
        up = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_LANCZOS4)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(up)
        _, bw = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 'metres' 라인을 대상으로만 OCR
        cfg = "--oem 3 --psm 6 -l eng -c tessedit_char_blacklist=ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        data = pytesseract.image_to_data(bw, config=cfg, output_type=pytesseract.Output.DICT)

        best_val = None
        best_conf = -1.0
        n = len(data["text"])
        for i in range(n):
            t = data["text"][i].lower().strip()
            if t in ("metre", "metres"):
                # 왼쪽으로 인접한 숫자 토큰 찾기
                # (같은 라인만 허용, 글자 높이로 너무 작은거 제거)
                line_i = data["line_num"][i]
                for j in range(i - 1, -1, -1):
                    if data["line_num"][j] != line_i: break
                    tj = data["text"][j].lower().strip()
                    if not tj: continue
                    # 흔한 오인식 교정
                    tj = tj.replace("o.", "0.").replace(",", ".")
                    # 소수만 허용(정수 단독 금지)
                    m = re.fullmatch(r"\d+\.\d+", tj)
                    if m:
                        h_box = data["height"][j]
                        if h_box < 12:  # 너무 작은 글자는 버림
                            continue
                        conf = float(data["conf"][j]) if data["conf"][j] != '-1' else 60.0
                        val = float(m.group(0))
                        if conf > best_conf:
                            best_conf = conf
                            best_val = val
                        break  # 가장 가까운 숫자만 사용

        # 유닛을 못 찾았으면 실패
        if best_val is None:
            return float("nan")

        # 이상치 필터: 프레임 간 급격한 점프 제거
        prev = getattr(self, "prev_dist", np.nan)
        if not np.isnan(prev):
            if abs(best_val - prev) > 1.0:  # 한 프레임에 1m 이상 점프면 버림
                return float("nan")

        self.prev_dist = best_val
        return best_val


if __name__ == '__main__':
    env = QWOPEnv()
    from time import time
    t0 = time()
    screenshot = env.get_state(4)
    t1 = time()
    print(f"Time taken: {t1 - t0:.3f} seconds")
    print(screenshot.shape)
    for i in range(4):
        Image.fromarray(screenshot[i]).save(f"array_screenshot_{i}.png")
    sleep(2)
