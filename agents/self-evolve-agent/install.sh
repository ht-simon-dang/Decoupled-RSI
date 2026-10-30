#!/usr/bin/env bash
set -euo pipefail
ENTRY_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
case "${1:-install}" in
    --help|-h|help)
        [[ $# -le 1 ]] || exit 2
        printf '%s\n' 'Usage: ./install.sh [--help]' \
            'With no arguments, install the Drug Agent runtime used by Self-Evolve.' \
            'Optional models, scientific backends, and the private regression suite are separate.'
        exit 0
        ;;
    install) [[ $# -eq 0 ]] || { printf '%s\n' 'Use no arguments to install, or --help.' >&2; exit 2; } ;;
    *) printf '%s\n' 'Use no arguments to install, or --help.' >&2; exit 2 ;;
esac
exec bash "$ENTRY_DIR/../drug-agent/install.sh" core
