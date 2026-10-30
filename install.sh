#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    cat <<'HELP'
Usage: ./install.sh COMMAND [ARGS...]

  agent drug [ARGS...]          Install the primary runtime (default: core).
  agent peptide [ARGS...]       Run the experimental Peptide Agent installer.
  agent mrna [ARGS...]          Run the experimental mRNA Agent installer.
  agent enzyme [ARGS...]        Run the experimental Enzyme Agent installer.
  agent self-evolve [ARGS...]   Install the runtime used by the Self-Evolve entrypoint.
  dependency FAMILY [ARGS...]   Select drug, peptide, mrna, or enzyme dependency installer.
  core                         Alias for "agent drug core".
  pfm | pfm-source              Forward a PFM installation stage to Drug Agent.
  external NAME | side-env NAME Forward a selected Drug Agent dependency stage.
  list                         List the Drug Agent upstream checkouts.
  --help                       Show this help without installing anything.

Examples:
  ./install.sh agent drug --help
  ./install.sh agent drug core
  ./install.sh dependency drug --help
  ./install.sh dependency peptide --help

All Agent projects live under agents/. Review THIRD_PARTY.md and the selected
dependency manifest before running an installation. Help and planning commands
do not download models. Installation requires a checkout path without a colon.
HELP
}

command_name="${1:---help}"
case "$command_name" in
    --help|-h|help) usage; exit 0 ;;
    core|pfm|pfm-source|external|side-env|list)
        exec bash "$REPO_DIR/agents/drug-agent/install.sh" "$@"
        ;;
    agent)
        [[ $# -ge 2 ]] || { usage >&2; exit 2; }
        agent_name="$2"
        shift 2
        case "$agent_name" in
            drug|drug-agent)
                [[ $# -gt 0 ]] || set -- core
                exec bash "$REPO_DIR/agents/drug-agent/install.sh" "$@"
                ;;
            peptide|peptide-agent) agent_dir=peptide-agent ;;
            mrna|mRNA-agent) agent_dir=mRNA-agent ;;
            enzyme|Enzyme-Agent) agent_dir=Enzyme-Agent ;;
            self-evolve|self-evolve-agent) agent_dir=self-evolve-agent ;;
            *) printf 'Unknown Agent: %s\n' "$agent_name" >&2; exit 2 ;;
        esac
        exec bash "$REPO_DIR/agents/$agent_dir/install.sh" "$@"
        ;;
    dependency|dependencies)
        [[ $# -ge 2 ]] || { usage >&2; exit 2; }
        family="$2"
        shift 2
        case "$family" in
            drug|peptide|mrna|enzyme) ;;
            *) printf 'Unknown dependency family: %s\n' "$family" >&2; exit 2 ;;
        esac
        exec bash "$REPO_DIR/third_party/$family/install.sh" "$@"
        ;;
    *) usage >&2; exit 2 ;;
esac
