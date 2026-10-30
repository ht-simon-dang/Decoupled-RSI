#!/usr/bin/env bash
set -euo pipefail
stage=core
if [ "$#" -gt 0 ]; then
    case "$1" in
        --help|-h|help)
            printf 'Usage: bootstrap.sh [--stage runtime|core]\n'
            exit 0
            ;;
        --stage)
            if [ "$#" -ne 2 ] || { [ "$2" != runtime ] && [ "$2" != core ]; }; then
                printf 'Expected --stage runtime or --stage core.\n' >&2
                exit 2
            fi
            stage="$2"
            ;;
        *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
    esac
fi
enzyme_project="$(cd -- "$(dirname -- "$0")/.." && pwd)"
enzyme_data="${ENZYME_DATA_DIR:-$HOME/.cache/decoupled-rsi/enzyme}"
enzyme_base_python="${PEPTIDE_PYTHON:-$enzyme_project/../peptide-agent/.venv/bin/python}"
if [ ! -x "$enzyme_base_python" ]; then
    printf 'Install peptide-agent first, or set PEPTIDE_PYTHON to its Python interpreter.\n' >&2
    exit 2
fi
"$enzyme_base_python" - <<'PY'
import torch,transformers,pydantic,fastapi,numpy,Bio,sklearn,lmformatenforcer,hatchling
assert transformers.__version__=='5.20.1', 'Use the verified peptide-agent Transformers 5.20.1 runtime'
assert tuple(map(int,torch.__version__.split('+')[0].split('.')[:2])) >= (2,14)
print('Verified base runtime:',torch.__version__,transformers.__version__)
PY
mkdir -p -- "$enzyme_data"
"$enzyme_base_python" -m venv --system-site-packages "$enzyme_data/env"
# A venv created from another venv does not necessarily inherit that venv's
# packages. Ensure the declared base stack exists in the destination as well.
"$enzyme_data/env/bin/python" -m pip install --no-cache-dir -r "$enzyme_project/../peptide-agent/requirements.txt"
"$enzyme_data/env/bin/python" -m pip install --no-cache-dir --no-deps -r "$enzyme_project/requirements.txt"
"$enzyme_data/env/bin/python" -m pip install -e "$enzyme_project" --no-deps --no-build-isolation
if [ "$stage" = runtime ]; then
    printf 'Runtime installed; model assets are separate components.\n'
    exit 0
fi
"$enzyme_data/env/bin/python" -m enzyme_agent.cli download core
"$enzyme_data/env/bin/python" "$enzyme_project/scripts/prepare_dnase.py"
"$enzyme_data/env/bin/python" -m enzyme_agent.cli doctor
