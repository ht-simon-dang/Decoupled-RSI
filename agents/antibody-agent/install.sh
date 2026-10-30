#!/usr/bin/env bash
set -euo pipefail
antibody_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export UV_CACHE_DIR="$antibody_root/.work/uv-cache"
export PIP_CACHE_DIR="$antibody_root/.work/pip-cache"
antibody_extra="${1:-base}"
case "$antibody_extra" in
  base) antibody_package="$antibody_root" ;;
  llm|test|mcp) antibody_package="$antibody_root[$antibody_extra]" ;;
  *) echo "Usage: $0 [base|llm|test|mcp]" >&2; exit 2 ;;
esac
if command -v uv >/dev/null 2>&1; then
  if [[ ! -x "$antibody_root/.venv/bin/python" ]]; then
    uv venv --python "${ANTIBODY_BOOTSTRAP_PYTHON:-3.11}" "$antibody_root/.venv"
  fi
  uv pip install --python "$antibody_root/.venv/bin/python" -e "$antibody_package"
else
  "${ANTIBODY_BOOTSTRAP_PYTHON:-python3}" -m venv "$antibody_root/.venv"
  "$antibody_root/.venv/bin/python" -m pip install -e "$antibody_package"
fi
echo "Installed. Launch: $antibody_root/run.sh serve --port 2007"
