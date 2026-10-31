#!/bin/zsh
set -e

cd -- "$(dirname -- "$0")"
if [[ ! -x .venv/bin/python ]]; then
  echo "Python environment missing. Run: python -m venv .venv && .venv/bin/python -m pip install -e ."
  read -r '?Press Enter to close...'
  exit 1
fi

.venv/bin/python scripts/run_bulk_spxw.py
open results/bulk/latest/dashboard.html
