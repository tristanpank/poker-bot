#!/usr/bin/env bash

set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/phone-tunnels-common.sh"

if [[ "${1-}" == "-h" || "${1-}" == "--help" ]]; then
    cat <<'EOF'
Usage: ./scripts/Stop-PhoneTunnels.sh
EOF
    exit 0
fi

cd "$PHONE_TUNNELS_REPO_ROOT"

phone_tunnels_stop_managed_processes "$PHONE_TUNNELS_STATE_PATH"
phone_tunnels_docker_compose stop frontend backend redis

phone_tunnels_log "Stopped phone tunnel services."
