#!/usr/bin/env bash
# Install only the component explicitly requested by the operator.
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONDA_BIN="${CONDA_EXE:-conda}"
ENV_NAME="${DRUGAGENT_CONDA_ENV:-drug-agent}"
EXTERNAL_DIR="${DRUGAGENT_EXTERNAL_DIR:-$SCRIPT_DIR/.external}"

usage() {
    cat <<'HELP'
Usage: ./install.sh COMMAND [NAME]

  core                 Create/update the main runtime environment and install drugagent.
  pfm                  Install PFM dependencies into the main environment; may compile.
  pfm-source           Fetch the pinned PFM upstream checkout into .external/PFM.
  external NAME        Fetch one upstream checkout listed by "list"; no environment install.
  side-env NAME        Create/update one optional environment:
                       aizynth, boltz, synflownet, unimol, vina
  list                 List known upstream checkout names.
  --help               Show this message without installing anything.

Examples:
  ./install.sh core
  ./install.sh pfm
  ./install.sh pfm-source
  ./install.sh external REINVENT4
  ./install.sh side-env vina

Requirements: conda; git for source checkout commands. Installation needs network.
Optional checkpoints and tool-specific setup remain separate from the main install.
Git LFS downloads are disabled during checkout; fetch needed LFS assets explicitly.
The three experimental Task Agents have their own installation entrypoints.

Environment overrides:
  CONDA_EXE                 Path to the conda executable.
  DRUGAGENT_CONDA_ENV        Main environment name (default: drug-agent).
  DRUGAGENT_EXTERNAL_DIR     Source-checkout directory (default: .external).

For execution, use a checkout path without a colon (:), because the candidate
worker uses PYTHONPATH. The publication staging title may contain a colon.
The private regression suite is not distributed; the unchanged self-evolution
gate 0 refuses to pass when that suite is missing.
HELP
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || {
        printf 'Required command is unavailable: %s\n' "$1" >&2
        exit 1
    }
}

require_runtime_path() {
    if [[ "$SCRIPT_DIR" == *:* ]]; then
        printf '%s\n' 'Use a checkout directory without ":" before installation or execution (for example decoupled-rsi-agent).' >&2
        exit 2
    fi
}

install_environment() {
    local env_file="$1" env_name="$2"
    require_command "$CONDA_BIN"
    if "$CONDA_BIN" env list | awk -v name="$env_name" '$1 == name { found=1 } END { exit !found }'; then
        "$CONDA_BIN" env update --name "$env_name" --file "$env_file"
    else
        "$CONDA_BIN" env create --name "$env_name" --file "$env_file"
    fi
}

clone_checkout() {
    local name="$1" url="$2" revision="$3" destination="$EXTERNAL_DIR/$1"
    require_command git
    if [[ -e "$destination" ]]; then
        local current
        current="$(git -C "$destination" rev-parse HEAD 2>/dev/null)" || {
            printf 'Existing destination is not a usable Git checkout: %s\n' "$destination" >&2
            exit 1
        }
        if [[ "$current" == "$revision"* ]]; then
            printf 'Pinned checkout already present: %s (%s)\n' "$name" "$current"
            return
        fi
        printf 'Existing %s is at %s; expected %s. Review it manually before replacing it.\n' "$name" "$current" "$revision" >&2
        exit 1
    fi
    mkdir -p "$EXTERNAL_DIR"
    GIT_LFS_SKIP_SMUDGE=1 git clone --no-checkout "$url" "$destination"
    GIT_LFS_SKIP_SMUDGE=1 git -C "$destination" checkout --detach "$revision"
    printf 'Fetched %s at %s. Follow its dependency and checkpoint instructions before use.\n' "$name" "$revision"
}

command_name="${1:---help}"
case "$command_name" in
    --help|-h|help)
        usage
        exit 0
        ;;
    list)
        awk -F '\t' '!/^#/ && NF >= 3 { print $1 }' "$SCRIPT_DIR/scripts/external/CHECKOUTS.tsv"
        printf '%s\n' 'PFM (use pfm-source)'
        exit 0
        ;;
esac

require_runtime_path
cd "$SCRIPT_DIR"
case "$command_name" in
    core)
        install_environment environment.yml "$ENV_NAME"
        "$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" python -m pip install --no-deps -e .
        printf 'Installed drugagent. Activate with: conda activate %s\n' "$ENV_NAME"
        ;;
    pfm)
        require_command "$CONDA_BIN"
        "$CONDA_BIN" run -n "$ENV_NAME" python -c 'import torch'
        "$CONDA_BIN" env update --name "$ENV_NAME" --file environment-pfm.yml
        "$CONDA_BIN" run --no-capture-output -n "$ENV_NAME" python -m pip install --no-build-isolation \
            torch_scatter==2.1.2 torch_cluster==1.6.3
        printf '%s\n' 'PFM dependencies installed. Fetch the upstream checkout separately with ./install.sh pfm-source.'
        ;;
    pfm-source)
        clone_checkout PFM https://github.com/kurisu92725/PFM 33be6c1f3dc9f42deb7a27be974ab9f5fc241497
        ;;
    external)
        [[ $# -eq 2 ]] || { usage >&2; exit 2; }
        requested="$2"
        found=0
        while IFS=$'\t' read -r name url revision rest; do
            [[ -n "$name" && "$name" != \#* ]] || continue
            if [[ "$name" == "$requested" ]]; then
                clone_checkout "$name" "$url" "$revision"
                found=1
                break
            fi
        done < scripts/external/CHECKOUTS.tsv
        [[ "$found" -eq 1 ]] || { printf 'Unknown checkout: %s. Run ./install.sh list.\n' "$requested" >&2; exit 2; }
        ;;
    side-env)
        [[ $# -eq 2 ]] || { usage >&2; exit 2; }
        case "$2" in
            aizynth|boltz|synflownet|unimol|vina)
                install_environment "envs/drug-agent-$2.yml" "drug-agent-$2"
                ;;
            *)
                printf 'Unknown side environment: %s\n' "$2" >&2
                exit 2
                ;;
        esac
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
