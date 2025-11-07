# 학습 실행 스크립트
#!/usr/bin/env bash
set -euo pipefail

# 이 파일(envs/scripts/run_train.sh) 기준으로 프로젝트 루트 계산(두 단계 위)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# 패키지 찾게 PYTHONPATH 추가
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

# 모듈 방식 실행
python -m rl.train_dqn
