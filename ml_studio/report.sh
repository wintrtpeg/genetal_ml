#!/usr/bin/env bash
# 진단 리포트 수집. 무슨 일이 있어도 파일은 남는다.
cd "$(dirname "$0")"
PY=".venv/bin/python"
[ -x "$PY" ] || PY="$(command -v python3 || command -v python)"
if [ -z "$PY" ]; then
  echo "[X] 파이썬을 찾지 못했습니다."
  exit 1
fi
"$PY" scripts/collect_report.py "$@" 2>&1 | tee report_console.log
echo
echo "보내실 파일: diagnostic_report.txt  ·  report_console.log"
