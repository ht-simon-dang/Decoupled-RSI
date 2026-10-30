#!/usr/bin/env bash
set -euo pipefail
ENTRY_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -eq 0 || "${1:-}" == --help || "${1:-}" == -h ]]; then
    cat <<'HELP'
Usage: ./run.sh COMMAND [ARGS...]

Forwards commands to "drugagent self-evolve" from the Drug Agent project.
Examples: ./run.sh doctor; ./run.sh analyze; ./run.sh report CANDIDATE_ID

Requires the installed runtime. baseline/step/run retain the regression gate;
the private suite is not included in this public source snapshot.
CONDA_EXE and DRUGAGENT_CONDA_ENV select the conda executable and environment.
HELP
    exit 0
fi
PROJECT_DIR="$(CDPATH= cd -- "$ENTRY_DIR/../drug-agent" && pwd)"
if [[ "$PROJECT_DIR" == *:* ]]; then
    printf '%s\n' 'Use a checkout path without a colon for evolution subprocesses.' >&2
    exit 2
fi
cd "$PROJECT_DIR"
exec "${CONDA_EXE:-conda}" run --no-capture-output -n "${DRUGAGENT_CONDA_ENV:-drug-agent}" \
    drugagent self-evolve "$@"
