#!/usr/bin/env bash
# 첫 실행. 실제 일은 scripts/setup.py 가 한다 (윈도우와 같은 코드).
cd "$(dirname "$0")"
PY=""
for c in python3 python; do
  command -v "$c" >/dev/null 2>&1 && PY="$c" && break
done
if [ -z "$PY" ]; then
  echo "[X] 파이썬을 찾지 못했습니다. 3.10 이상을 설치해 주세요."
  exit 1
fi
# 첫 실행은 설치·테스트까지, 두 번째부터는 화면만 띄운다.
#   ./run.sh          평소 실행
#   ./run.sh --full   설치와 점검을 다시
#   ./run.sh 3.12     그 버전으로 (첫 실행에만 의미가 있다)
case "$1" in
  "")   exec "$PY" scripts/setup.py ;;
  -*)   exec "$PY" scripts/setup.py "$@" ;;
  *)    exec "$PY" scripts/setup.py --python "$1" ;;
esac
