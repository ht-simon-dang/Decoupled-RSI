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

With no arguments, install the mRNA runtime, pinned upstream code and model
assets. This needs network access, conda, build tools, and substantial storage.

PEPTIDE_PYTHON selects the base interpreter (default: sibling peptide .venv).
MRNA_DATA_DIR selects the data root (default: ~/.cache/decoupled-rsi/mrna).
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
mrna_bootstrap_python="${PEPTIDE_PYTHON:-$project_dir/../peptide-agent/.venv/bin/python}"
cd -- "$project_dir"
if [ ! -x "$mrna_bootstrap_python" ]; then
    printf 'Install peptide-agent first, or set PEPTIDE_PYTHON to its Python interpreter.\n' >&2
    exit 2
fi
export PEPTIDE_PYTHON="$mrna_bootstrap_python"
exec "$mrna_bootstrap_python" scripts/bootstrap.py "$@"
