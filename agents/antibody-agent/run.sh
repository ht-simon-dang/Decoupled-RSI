#!/usr/bin/env bash
set -euo pipefail
antibody_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Trusted, ignored, package-local machine settings (never read the repository-wide runtime file).
if [[ -f "$antibody_root/.antibody-agent/runtime.env" ]]; then
  source "$antibody_root/.antibody-agent/runtime.env"
fi
antibody_python="${ANTIBODY_PYTHON:-$antibody_root/.venv/bin/python}"
if [[ ! -x "$antibody_python" ]]; then
  echo "Run agents/antibody-agent/install.sh first, or set ANTIBODY_PYTHON." >&2
  exit 2
fi
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$antibody_root/src${PYTHONPATH:+:$PYTHONPATH}"
export ANTIBODY_DATA_DIR="${ANTIBODY_DATA_DIR:-$antibody_root/.antibody-agent}"
exec "$antibody_python" -m antibody_agent "$@"
