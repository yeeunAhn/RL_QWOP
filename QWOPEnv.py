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

    def get_state(self, n: int) -> np.ndarray:
        np_list = []
        for i in range(n):
            with mss() as sct:
                screenshot = sct.grab(self.game_obj_location)
            array_2d = np.array(screenshot)[:, :, 0]
            print(array_2d.shape)
            np_list.append(array_2d)
        return np.stack(np_list, axis=0)

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
