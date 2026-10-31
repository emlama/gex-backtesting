#!/bin/zsh
set -e

cd -- "$(dirname -- "$0")"
if [[ ! -x .venv/bin/python ]]; then
  echo "缺少 Python 環境。請先執行：python3 -m venv .venv && .venv/bin/python -m pip install -e ."
  read -r '?按 Enter 關閉…'
  exit 1
fi

.venv/bin/python scripts/uw_workbench.py
