#!/usr/bin/env bash
set -euo pipefail
installer_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
exec "${DEPENDENCY_PYTHON:-python3}" "$installer_dir/manage.py" "$@"
