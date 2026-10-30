#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "$0")/.." && pwd)"
enzyme_data="${ENZYME_DATA_DIR:-$HOME/.cache/decoupled-rsi/enzyme}"
enzyme_python="${ENZYME_PYTHON:-$enzyme_data/env/bin/python}"
foundry_base="${ENZYME_FOUNDRY_BASE:?Set ENZYME_FOUNDRY_BASE to a Python 3.12 interpreter with CUDA Torch}"
"$enzyme_python" -c 'from enzyme_agent.config import check_space; check_space(6000000000)'
"$enzyme_python" -m enzyme_agent.extended_models rfdiffusion3 thermompnn
"$foundry_base" -c 'import sys,torch; assert sys.version_info[:2]==(3,12); assert torch.cuda.is_available()'
if [ ! -x "$enzyme_data/foundry-env/bin/python" ]; then
    "$foundry_base" -m venv --system-site-packages "$enzyme_data/foundry-env"
fi
# This venv only shadows inherited packages; the shared drug/peptide envs remain unchanged.
"$enzyme_data/foundry-env/bin/python" -m pip install --no-cache-dir 'rc-foundry[rfd3]==0.2.0' 'atomworks[ml]==3.0.1'
"$enzyme_data/foundry-env/bin/python" -m pip install --no-cache-dir --no-deps "$enzyme_data/models/rfdiffusion3/source"
"$enzyme_data/foundry-env/bin/python" -m pip freeze > "$enzyme_data/foundry-requirements.local.txt"
"$enzyme_python" -c 'from enzyme_agent.config import check_space; print("Enzyme bytes:",check_space())'
echo 'Extended assets installed. Local inference validation is required before readiness gates can pass.'
