import os
import numpy as np
from datetime import datetime
from pynput import keyboard

from envs.qwop_env import QWOPEnv, ACTIONS


# --- 키 입력 추적용 ---
pressed_keys = set()     # 현재 누르고 있는 키들 ('q','w','o','p')
last_action = 0          # 가장 최근 매칭된 action index


def find_action_from_keys(keys_set):
    """
    현재 사람이 누르고 있는 키(q,w,o,p) 조합을 보고 ACTION index 반환.
    """
    key_list = sorted(list(keys_set))

    for idx, combo in ACTIONS.items():
        if sorted(combo) == key_list:
            return idx
    return 0  # 아무 매칭 없는 경우 no-op


def on_press(key):
    global last_action

    try:
        k = key.char.lower()
    except:
        return

    if k in ['q', 'w', 'o', 'p']:
        pressed_keys.add(k)
        last_action = find_action_from_keys(pressed_keys)


def on_release(key):
    global last_action

    try:
        k = key.char.lower()
    except:
        return

    if k in ['q', 'w', 'o', 'p']:
        if k in pressed_keys:
            pressed_keys.remove(k)
        last_action = find_action_from_keys(pressed_keys)


def main():
    print("=== Human Keyboard QWOP Collector (episode별 저장) ===")
    print("키보드:")
    print("   Q / W / O / P : 실제로 눌러서 조합 만들기")
    print("   CTRL+C        : 프로그램 종료\n")

    # 저장 폴더 (원하면 바꿔도 됨)
    base_dir = os.path.join(os.getcwd(), "human_episodes")
    os.makedirs(base_dir, exist_ok=True)
    print(f"저장 폴더: {base_dir}")

    env = QWOPEnv(debug_ocr=False, frame_stack=1, background_safe=True, debug_posture=False)

    # --- 키보드 리스너 시작 ---
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    episode = 0

    try:
        while True:
            obs = env.reset()
            done = False
            step = 0

            # 에피소드 버퍼 (이 에피소드 것만 모음)
            ep_states = []
            ep_actions = []
            ep_rewards = []
            ep_next_states = []
            ep_dones = []
            ep_distances = []

            print(f"\n=== Episode {episode} 시작 ===")

            while not done:
                action = last_action   # 사람이 현재 누르고 있는 키 조합 → action index

                state = obs.copy()
                obs_next, reward, done, info = env.step(action)
                dist = info.get("distance", float("nan"))

                dist_s = "?.??" if np.isnan(dist) else f"{dist:.2f}"
                print(f"[Ep {episode} | Step {step}] act={action} keys={ACTIONS[action]} dist={dist_s} r={reward:.3f}")

                ep_states.append(state)
                ep_actions.append(action)
                ep_rewards.append(reward)
                ep_next_states.append(obs_next)
                ep_dones.append(done)
                ep_distances.append(dist)

                obs = obs_next
                step += 1

            # ----- 에피소드 종료: 이 에피소드만 npz로 저장 -----
            if len(ep_states) > 0:
                states_arr = np.stack(ep_states, axis=0)
                next_states_arr = np.stack(ep_next_states, axis=0)
                actions_arr = np.array(ep_actions, dtype=np.int64)
                rewards_arr = np.array(ep_rewards, dtype=np.float32)
                dones_arr = np.array(ep_dones, dtype=bool)
                distances_arr = np.array(ep_distances, dtype=np.float32)

                # 마지막 거리 (잘한/못한 회차 구분용)
                final_dist = distances_arr[-1] if len(distances_arr) > 0 else np.nan

                # 여기서 threshold에 따라 good/bad 접두사 붙이고 싶으면:
                # GOOD_THRESH = 10.0
                # tag = "good" if final_dist >= GOOD_THRESH else "bad"
                # filename = f"{tag}_ep{episode:04d}_dist{final_dist:.2f}.npz"

                filename = f"ep{episode:04d}_dist{final_dist:.2f}.npz"
                save_path = os.path.join(base_dir, filename)

                np.savez_compressed(
                    save_path,
                    states=states_arr,
                    actions=actions_arr,
                    rewards=rewards_arr,
                    next_states=next_states_arr,
                    dones=dones_arr,
                    distances=distances_arr,
                )

                print(f"✅ Episode {episode} 저장 완료: {save_path} (steps={len(ep_states)}, final_dist={final_dist:.2f})")
            else:
                print(f"⚠ Episode {episode}: 저장할 transition 없음")

            episode += 1

    except KeyboardInterrupt:
        print("\n[CTRL+C] 종료됨.")

    finally:
        listener.stop()
        env.close()
        print("환경 종료.")


if __name__ == "__main__":
    main()
