#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
enzyme_runtime_root="${ENZYME_DATA_DIR:-$HOME/.cache/decoupled-rsi/enzyme}"
enzyme_python="${ENZYME_PYTHON:-$enzyme_runtime_root/env/bin/python}"
exec "$enzyme_python" -m enzyme_agent.cli "$@"
