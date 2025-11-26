import torch
import os
import numpy as np

# ==========================================
# 👇 확인하고 싶은 체크포인트 파일 경로를 여기에 적으세요
CHECKPOINT_PATH = "checkpoints/qwop_checkpoint_20251120_181519_steps35000.pth"


# ==========================================

def inspect_checkpoint(path):
    if not os.path.exists(path):
        print(f"❌ 오류: 파일을 찾을 수 없습니다 -> {path}")
        return

    print(f"\n🔍 모델 가중치 분석 중... [{path}]")
    print("=" * 60)

    try:
        # CPU로 로드 (GPU가 없어도 확인 가능하게)
        state_dict = torch.load(path, map_location=torch.device('cpu'))

        total_params = 0

        for layer_name, tensor in state_dict.items():
            # 텐서를 numpy 배열로 변환
            param = tensor.numpy()
            shape = param.shape
            size = param.size
            total_params += size

            # 통계 계산
            mean_val = np.mean(param)
            std_val = np.std(param)
            min_val = np.min(param)
            max_val = np.max(param)

            print(f"LAYER: {layer_name}")
            print(f" • 형태(Shape): {shape}")
            print(f" • 파라미터 수: {size:,}")
            print(f" • 통계: 평균={mean_val:.5f} | 표준편차={std_val:.5f}")
            print(f" • 범위: Min={min_val:.5f} ~ Max={max_val:.5f}")

            # 혹시 NaN(숫자 아님)이나 무한대(Inf)가 있는지 체크 (매우 중요!)
            if np.isnan(param).any():
                print(" 🚨 경고: NaN(망가진 값)이 발견되었습니다!")
            if np.isinf(param).any():
                print(" 🚨 경고: Inf(무한대)가 발견되었습니다!")

            print("-" * 60)

        print(f"✅ 분석 완료. 총 파라미터 개수: {total_params:,}")

    except Exception as e:
        print(f"❌ 파일 읽기 실패: {e}")


if __name__ == "__main__":
    inspect_checkpoint(CHECKPOINT_PATH)