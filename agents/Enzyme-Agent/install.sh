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

With no arguments, install the Enzyme runtime and download its pinned core
models and public DNase reference. This needs network access and substantial
storage. Extended models have a separate installation entrypoint.

PEPTIDE_PYTHON selects the base interpreter (default: sibling peptide .venv).
ENZYME_DATA_DIR selects the data root (default: ~/.cache/decoupled-rsi/enzyme).
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
exec bash "$project_dir/scripts/bootstrap.sh" "$@"
