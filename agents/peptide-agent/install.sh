#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -gt 0 ]; then
    case "$1" in
        --help|-h|help)
            if [ "$#" -ne 1 ]; then
                printf 'Help accepts no additional arguments.\n' >&2
                exit 2
            fi
            cat <<'USAGE'
Usage: bash install.sh [--help|-h|help]

With no arguments, create a Python 3.11 virtual environment and install the
peptide-agent runtime. This needs network access; model weights are separate.

PEPTIDE_BOOTSTRAP_PYTHON selects the base interpreter (default: python3.11).
PEPTIDE_ENV_DIR selects the environment directory (default: project/.venv).
USAGE
            exit 0
            ;;
        *)
            printf 'Unknown argument: %s\nUse --help for usage.\n' "$1" >&2
            exit 2
            ;;
    esac
fi
project_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
peptide_bootstrap_python="${PEPTIDE_BOOTSTRAP_PYTHON:-python3.11}"
peptide_env_dir="${PEPTIDE_ENV_DIR:-$project_dir/.venv}"
cd -- "$project_dir"
if [ ! -x "$peptide_env_dir/bin/python" ]; then
    "$peptide_bootstrap_python" -m venv "$peptide_env_dir"
fi
"$peptide_env_dir/bin/python" -m pip install -r requirements.txt
"$peptide_env_dir/bin/python" -m pip install -e . --no-deps
printf 'Runtime installed. Model weights are separate; see README.md.\n'
