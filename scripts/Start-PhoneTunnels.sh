#!/usr/bin/env bash

set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/phone-tunnels-common.sh"

usage() {
    cat <<'EOF'
Usage: ./scripts/Start-PhoneTunnels.sh [options]

Options:
  --frontend-local-url URL
  --backend-local-url URL
  --short-path PATH
  --shortio-api-key KEY
  --shortio-domain DOMAIN
  --shortio-link-id ID
  -h, --help
EOF
}

FRONTEND_LOCAL_URL_EXPLICIT=""
BACKEND_LOCAL_URL_EXPLICIT=""
SHORT_PATH_EXPLICIT=""
SHORTIO_API_KEY_EXPLICIT=""
SHORTIO_DOMAIN_EXPLICIT=""
SHORTIO_LINK_ID_EXPLICIT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --frontend-local-url)
            FRONTEND_LOCAL_URL_EXPLICIT="${2-}"
            shift 2
            ;;
        --backend-local-url)
            BACKEND_LOCAL_URL_EXPLICIT="${2-}"
            shift 2
            ;;
        --short-path)
            SHORT_PATH_EXPLICIT="${2-}"
            shift 2
            ;;
        --shortio-api-key)
            SHORTIO_API_KEY_EXPLICIT="${2-}"
            shift 2
            ;;
        --shortio-domain)
            SHORTIO_DOMAIN_EXPLICIT="${2-}"
            shift 2
            ;;
        --shortio-link-id)
            SHORTIO_LINK_ID_EXPLICIT="${2-}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            phone_tunnels_die "Unknown argument: $1"
            ;;
    esac
done

env_frontend_local_url="${FRONTEND_LOCAL_URL-}"
env_backend_local_url="${BACKEND_LOCAL_URL-}"
env_short_path="${SHORTIO_PATH-}"
env_shortio_api_key="${SHORTIO_API_KEY-}"
env_shortio_domain="${SHORTIO_DOMAIN-}"
env_shortio_link_id="${SHORTIO_LINK_ID-}"

phone_tunnels_require_command curl
phone_tunnels_ensure_python
phone_tunnels_ensure_state_dir
phone_tunnels_load_config

FRONTEND_LOCAL_URL="$(phone_tunnels_preferred_value "$FRONTEND_LOCAL_URL_EXPLICIT" "${FRONTEND_LOCAL_URL-}" "$env_frontend_local_url" "http://localhost:3000")"
BACKEND_LOCAL_URL="$(phone_tunnels_preferred_value "$BACKEND_LOCAL_URL_EXPLICIT" "${BACKEND_LOCAL_URL-}" "$env_backend_local_url" "http://localhost:8000")"
SHORT_PATH="$(phone_tunnels_preferred_value "$SHORT_PATH_EXPLICIT" "${SHORTIO_PATH-}" "$env_short_path" "poker")"
SHORTIO_API_KEY="$(phone_tunnels_preferred_value "$SHORTIO_API_KEY_EXPLICIT" "${SHORTIO_API_KEY-}" "$env_shortio_api_key" "")"
SHORTIO_DOMAIN="$(phone_tunnels_preferred_value "$SHORTIO_DOMAIN_EXPLICIT" "${SHORTIO_DOMAIN-}" "$env_shortio_domain" "")"
SHORTIO_LINK_ID="$(phone_tunnels_preferred_value "$SHORTIO_LINK_ID_EXPLICIT" "${SHORTIO_LINK_ID-}" "$env_shortio_link_id" "")"

existing_short_provider="$(phone_tunnels_json_get "$PHONE_TUNNELS_STATE_PATH" "shortUrl.provider")"
existing_short_link_id="$(phone_tunnels_json_get "$PHONE_TUNNELS_STATE_PATH" "shortUrl.linkId")"
existing_short_url="$(phone_tunnels_json_get "$PHONE_TUNNELS_STATE_PATH" "shortUrl.url")"
existing_short_path="$(phone_tunnels_json_get "$PHONE_TUNNELS_STATE_PATH" "shortUrl.path")"
existing_short_domain="$(phone_tunnels_json_get "$PHONE_TUNNELS_STATE_PATH" "shortUrl.domain")"

resolved_shortio_link_id="$SHORTIO_LINK_ID"
if [[ -z "$resolved_shortio_link_id" ]]; then
    resolved_shortio_link_id="$existing_short_link_id"
fi

resolved_shortio_domain="$SHORTIO_DOMAIN"
if [[ -z "$resolved_shortio_domain" ]]; then
    resolved_shortio_domain="$existing_short_domain"
fi

resolved_short_path="$SHORT_PATH"
if [[ -z "$resolved_short_path" ]]; then
    resolved_short_path="$existing_short_path"
fi

cd "$PHONE_TUNNELS_REPO_ROOT"

cloudflared_path="$(phone_tunnels_get_cloudflared_command)"
phone_tunnels_stop_managed_processes "$PHONE_TUNNELS_STATE_PATH"

phone_tunnels_log "Starting backend service..."
phone_tunnels_docker_compose up -d backend

phone_tunnels_log "Opening backend Quick Tunnel..."
phone_tunnels_start_quick_tunnel "backend" "$BACKEND_LOCAL_URL" "$cloudflared_path" "$PHONE_TUNNELS_STATE_DIR"
backend_tunnel_url="$PHONE_TUNNELS_LAST_TUNNEL_URL"
backend_tunnel_pid="$PHONE_TUNNELS_LAST_TUNNEL_PID"
backend_tunnel_stdout_path="$PHONE_TUNNELS_LAST_TUNNEL_STDOUT_PATH"
backend_tunnel_stderr_path="$PHONE_TUNNELS_LAST_TUNNEL_STDERR_PATH"

phone_tunnels_log "Starting frontend service with NEXT_PUBLIC_BACKEND_URL=$backend_tunnel_url"
NEXT_PUBLIC_BACKEND_URL="$backend_tunnel_url" phone_tunnels_docker_compose up -d --force-recreate frontend

phone_tunnels_log "Opening frontend Quick Tunnel..."
phone_tunnels_start_quick_tunnel "frontend" "$FRONTEND_LOCAL_URL" "$cloudflared_path" "$PHONE_TUNNELS_STATE_DIR"
frontend_tunnel_url="$PHONE_TUNNELS_LAST_TUNNEL_URL"
frontend_tunnel_pid="$PHONE_TUNNELS_LAST_TUNNEL_PID"
frontend_tunnel_stdout_path="$PHONE_TUNNELS_LAST_TUNNEL_STDOUT_PATH"
frontend_tunnel_stderr_path="$PHONE_TUNNELS_LAST_TUNNEL_STDERR_PATH"

final_short_provider="$existing_short_provider"
final_short_link_id="$existing_short_link_id"
final_short_url="$existing_short_url"
final_short_path="$existing_short_path"
final_short_domain="$existing_short_domain"
fresh_short_url=""

if [[ -n "$SHORTIO_API_KEY" ]]; then
    phone_tunnels_log "Updating Short.io short URL..."
    phone_tunnels_set_shortio_short_url "$SHORTIO_API_KEY" "$frontend_tunnel_url" "$resolved_shortio_domain" "$resolved_short_path" "$resolved_shortio_link_id"
    fresh_short_url="$PHONE_TUNNELS_LAST_SHORT_URL"
    final_short_provider="short.io"
    final_short_link_id="$PHONE_TUNNELS_LAST_SHORT_LINK_ID"
    final_short_url="$PHONE_TUNNELS_LAST_SHORT_URL"
    final_short_path="$resolved_short_path"
    final_short_domain="$resolved_shortio_domain"
fi

PHONE_TUNNELS_BACKEND_URL="$backend_tunnel_url" \
PHONE_TUNNELS_BACKEND_PID="$backend_tunnel_pid" \
PHONE_TUNNELS_BACKEND_STDOUT_PATH="$backend_tunnel_stdout_path" \
PHONE_TUNNELS_BACKEND_STDERR_PATH="$backend_tunnel_stderr_path" \
PHONE_TUNNELS_FRONTEND_URL="$frontend_tunnel_url" \
PHONE_TUNNELS_FRONTEND_PID="$frontend_tunnel_pid" \
PHONE_TUNNELS_FRONTEND_STDOUT_PATH="$frontend_tunnel_stdout_path" \
PHONE_TUNNELS_FRONTEND_STDERR_PATH="$frontend_tunnel_stderr_path" \
PHONE_TUNNELS_SHORT_PROVIDER="$final_short_provider" \
PHONE_TUNNELS_SHORT_LINK_ID="$final_short_link_id" \
PHONE_TUNNELS_SHORT_URL="$final_short_url" \
PHONE_TUNNELS_SHORT_PATH="$final_short_path" \
PHONE_TUNNELS_SHORT_DOMAIN="$final_short_domain" \
phone_tunnels_write_state "$PHONE_TUNNELS_STATE_PATH"

printf '\n'
phone_tunnels_log "Backend tunnel:  $backend_tunnel_url"
phone_tunnels_log "Frontend tunnel: $frontend_tunnel_url"
if [[ -n "$fresh_short_url" ]]; then
    phone_tunnels_log "Short URL:       $fresh_short_url"
elif [[ -n "$final_short_url" ]]; then
    phone_tunnels_log "Short URL:       $final_short_url"
else
    phone_tunnels_log "Short URL:       not updated (set SHORTIO_API_KEY and SHORTIO_DOMAIN for the first run)"
fi
printf '\n'
phone_tunnels_log "Tunnel state saved to $PHONE_TUNNELS_STATE_PATH"
