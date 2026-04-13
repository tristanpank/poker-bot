#!/usr/bin/env bash

PHONE_TUNNELS_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PHONE_TUNNELS_REPO_ROOT="$(cd -- "$PHONE_TUNNELS_SCRIPT_DIR/.." && pwd)"
PHONE_TUNNELS_STATE_DIR="$PHONE_TUNNELS_REPO_ROOT/.local/phone-tunnels"
PHONE_TUNNELS_CONFIG_PATH="$PHONE_TUNNELS_STATE_DIR/config.sh"
PHONE_TUNNELS_STATE_PATH="$PHONE_TUNNELS_STATE_DIR/state.json"

PHONE_TUNNELS_PYTHON=""
PHONE_TUNNELS_DOCKER_COMMAND=""
PHONE_TUNNELS_LAST_TUNNEL_NAME=""
PHONE_TUNNELS_LAST_TUNNEL_URL=""
PHONE_TUNNELS_LAST_TUNNEL_PID=""
PHONE_TUNNELS_LAST_TUNNEL_STDOUT_PATH=""
PHONE_TUNNELS_LAST_TUNNEL_STDERR_PATH=""
PHONE_TUNNELS_LAST_SHORT_LINK_ID=""
PHONE_TUNNELS_LAST_SHORT_URL=""

phone_tunnels_die() {
    echo "Error: $*" >&2
    exit 1
}

phone_tunnels_log() {
    echo "$*"
}

phone_tunnels_require_command() {
    local command_name="$1"
    command -v "$command_name" >/dev/null 2>&1 || phone_tunnels_die "$command_name is required but was not found on PATH."
}

phone_tunnels_get_docker_command() {
    local candidate
    local windows_candidates=(
        "docker.exe"
        "/c/Program Files/Docker/Docker/resources/bin/docker.exe"
        "/mnt/c/Program Files/Docker/Docker/resources/bin/docker.exe"
    )

    if [[ -n "$PHONE_TUNNELS_DOCKER_COMMAND" ]]; then
        printf '%s\n' "$PHONE_TUNNELS_DOCKER_COMMAND"
        return 0
    fi

    for candidate in "${windows_candidates[@]}"; do
        if command -v "$candidate" >/dev/null 2>&1; then
            PHONE_TUNNELS_DOCKER_COMMAND="$(command -v "$candidate")"
            printf '%s\n' "$PHONE_TUNNELS_DOCKER_COMMAND"
            return 0
        fi

        if [[ -x "$candidate" ]]; then
            PHONE_TUNNELS_DOCKER_COMMAND="$candidate"
            printf '%s\n' "$PHONE_TUNNELS_DOCKER_COMMAND"
            return 0
        fi
    done

    if command -v docker >/dev/null 2>&1; then
        PHONE_TUNNELS_DOCKER_COMMAND="$(command -v docker)"
        printf '%s\n' "$PHONE_TUNNELS_DOCKER_COMMAND"
        return 0
    fi

    phone_tunnels_die "docker is not installed or not visible to this shell."
}

phone_tunnels_ensure_python() {
    if [[ -n "$PHONE_TUNNELS_PYTHON" ]]; then
        return
    fi

    if command -v python3 >/dev/null 2>&1; then
        PHONE_TUNNELS_PYTHON="$(command -v python3)"
        return
    fi

    if command -v python >/dev/null 2>&1; then
        PHONE_TUNNELS_PYTHON="$(command -v python)"
        return
    fi

    phone_tunnels_die "python3 (or python) is required to read/write the tunnel state."
}

phone_tunnels_ensure_state_dir() {
    mkdir -p "$PHONE_TUNNELS_STATE_DIR"
}

phone_tunnels_get_os_name() {
    case "$(uname -s)" in
        Linux*)
            echo "Linux"
            ;;
        Darwin*)
            echo "OSX"
            ;;
        MINGW*|MSYS*|CYGWIN*|Windows_NT)
            echo "Windows"
            ;;
        *)
            echo "Unknown"
            ;;
    esac
}

phone_tunnels_docker_compose() {
    local compose_files=("docker-compose.yml")
    local os_name
    local file_args=()
    local file
    local docker_command

    os_name="$(phone_tunnels_get_os_name)"
    docker_command="$(phone_tunnels_get_docker_command)"
    if [[ "$os_name" == "Windows" && -f "docker-compose.windows.yml" ]]; then
        compose_files+=("docker-compose.windows.yml")
    elif [[ "$os_name" == "Linux" && -f "docker-compose.linux.yml" ]]; then
        compose_files+=("docker-compose.linux.yml")
    fi

    for file in "${compose_files[@]}"; do
        file_args+=("-f" "$file")
    done

    "$docker_command" compose "${file_args[@]}" "$@"
}

phone_tunnels_load_config() {
    if [[ -f "$PHONE_TUNNELS_CONFIG_PATH" ]]; then
        # shellcheck disable=SC1090
        source "$PHONE_TUNNELS_CONFIG_PATH"
    fi
}

phone_tunnels_preferred_value() {
    local explicit_value="${1-}"
    local config_value="${2-}"
    local environment_value="${3-}"
    local default_value="${4-}"

    if [[ -n "${explicit_value// }" ]]; then
        printf '%s\n' "$explicit_value"
        return
    fi

    if [[ -n "${config_value// }" ]]; then
        printf '%s\n' "$config_value"
        return
    fi

    if [[ -n "${environment_value// }" ]]; then
        printf '%s\n' "$environment_value"
        return
    fi

    printf '%s\n' "$default_value"
}

phone_tunnels_json_get() {
    local json_path="$1"
    local key_path="$2"

    [[ -f "$json_path" ]] || return 0
    phone_tunnels_ensure_python

    "$PHONE_TUNNELS_PYTHON" - "$json_path" "$key_path" <<'PY'
import json
import sys

json_path, key_path = sys.argv[1], sys.argv[2]

try:
    with open(json_path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
except Exception:
    sys.exit(0)

for key in key_path.split("."):
    if not isinstance(value, dict) or key not in value or value[key] is None:
        sys.exit(0)
    value = value[key]

if isinstance(value, (dict, list)):
    sys.stdout.write(json.dumps(value))
else:
    sys.stdout.write(str(value))
PY
}

phone_tunnels_write_state() {
    local state_path="$1"

    phone_tunnels_ensure_python

    "$PHONE_TUNNELS_PYTHON" - "$state_path" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone

state_path = sys.argv[1]
short_provider = os.environ.get("PHONE_TUNNELS_SHORT_PROVIDER", "").strip()

state = {
    "updatedAt": datetime.now(timezone.utc).isoformat(),
    "backend": {
        "name": "backend",
        "url": os.environ["PHONE_TUNNELS_BACKEND_URL"],
        "pid": int(os.environ["PHONE_TUNNELS_BACKEND_PID"]),
        "stdoutPath": os.environ["PHONE_TUNNELS_BACKEND_STDOUT_PATH"],
        "stderrPath": os.environ["PHONE_TUNNELS_BACKEND_STDERR_PATH"],
    },
    "frontend": {
        "name": "frontend",
        "url": os.environ["PHONE_TUNNELS_FRONTEND_URL"],
        "pid": int(os.environ["PHONE_TUNNELS_FRONTEND_PID"]),
        "stdoutPath": os.environ["PHONE_TUNNELS_FRONTEND_STDOUT_PATH"],
        "stderrPath": os.environ["PHONE_TUNNELS_FRONTEND_STDERR_PATH"],
    },
    "shortUrl": None,
}

if short_provider:
    state["shortUrl"] = {
        "provider": short_provider,
        "linkId": os.environ.get("PHONE_TUNNELS_SHORT_LINK_ID", ""),
        "url": os.environ.get("PHONE_TUNNELS_SHORT_URL", ""),
        "path": os.environ.get("PHONE_TUNNELS_SHORT_PATH", ""),
        "domain": os.environ.get("PHONE_TUNNELS_SHORT_DOMAIN", ""),
    }

with open(state_path, "w", encoding="utf-8") as handle:
    json.dump(state, handle, indent=2)
PY
}

phone_tunnels_stop_managed_processes() {
    local state_path="$1"
    local name
    local pid
    local command_line

    [[ -f "$state_path" ]] || return 0

    for name in backend frontend; do
        pid="$(phone_tunnels_json_get "$state_path" "$name.pid")"
        [[ -n "$pid" ]] || continue

        if ! kill -0 "$pid" 2>/dev/null; then
            continue
        fi

        command_line="$(ps -p "$pid" -o args= 2>/dev/null || true)"
        if [[ "$command_line" != *cloudflared* ]]; then
            continue
        fi

        kill "$pid" 2>/dev/null || true
        sleep 1
        if kill -0 "$pid" 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null || true
        fi

        phone_tunnels_log "Stopped $name cloudflared process ($pid)."
    done
}

phone_tunnels_get_cloudflared_command() {
    local candidate
    local windows_candidates=(
        "/c/Program Files (x86)/cloudflared/cloudflared.exe"
        "/c/Program Files/cloudflared/cloudflared.exe"
        "/mnt/c/Program Files (x86)/cloudflared/cloudflared.exe"
        "/mnt/c/Program Files/cloudflared/cloudflared.exe"
    )

    if command -v cloudflared >/dev/null 2>&1; then
        command -v cloudflared
        return 0
    fi

    for candidate in "${windows_candidates[@]}"; do
        if [[ -x "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done

    phone_tunnels_die "cloudflared is not installed or not visible to this shell. Install it first or add it to PATH."
}

phone_tunnels_start_quick_tunnel() {
    local name="$1"
    local local_url="$2"
    local cloudflared_path="$3"
    local log_dir="$4"
    local run_stamp
    local stdout_path
    local stderr_path
    local pid
    local timeout_at
    local url
    local path

    run_stamp="$(date -u +%Y%m%d-%H%M%S)-$$"
    stdout_path="$log_dir/$name-$run_stamp.log"
    stderr_path="$log_dir/$name-$run_stamp.err.log"

    "$cloudflared_path" tunnel --url "$local_url" --no-autoupdate >"$stdout_path" 2>"$stderr_path" &
    pid=$!
    timeout_at=$((SECONDS + 60))

    while (( SECONDS < timeout_at )); do
        sleep 0.5

        for path in "$stdout_path" "$stderr_path"; do
            [[ -f "$path" ]] || continue
            url="$(grep -Eo 'https://[-a-z0-9]+\.trycloudflare\.com' "$path" 2>/dev/null | head -n 1 || true)"
            if [[ -n "$url" ]]; then
                PHONE_TUNNELS_LAST_TUNNEL_NAME="$name"
                PHONE_TUNNELS_LAST_TUNNEL_URL="$url"
                PHONE_TUNNELS_LAST_TUNNEL_PID="$pid"
                PHONE_TUNNELS_LAST_TUNNEL_STDOUT_PATH="$stdout_path"
                PHONE_TUNNELS_LAST_TUNNEL_STDERR_PATH="$stderr_path"
                return 0
            fi
        done

        if ! kill -0 "$pid" 2>/dev/null; then
            phone_tunnels_die "cloudflared exited before publishing the $name tunnel URL. Check $stderr_path"
        fi
    done

    phone_tunnels_die "Timed out waiting for the $name tunnel URL. Check $stderr_path"
}

phone_tunnels_url_encode() {
    local raw_value="$1"

    phone_tunnels_ensure_python
    PHONE_TUNNELS_RAW_URL_VALUE="$raw_value" "$PHONE_TUNNELS_PYTHON" - <<'PY'
import os
import urllib.parse

print(urllib.parse.quote(os.environ["PHONE_TUNNELS_RAW_URL_VALUE"], safe=""))
PY
}

phone_tunnels_build_shortio_payload() {
    local long_url="$1"
    local domain="$2"
    local path="$3"
    local link_id="$4"

    phone_tunnels_ensure_python
    PHONE_TUNNELS_PAYLOAD_LONG_URL="$long_url" \
    PHONE_TUNNELS_PAYLOAD_DOMAIN="$domain" \
    PHONE_TUNNELS_PAYLOAD_PATH="$path" \
    PHONE_TUNNELS_PAYLOAD_LINK_ID="$link_id" \
    "$PHONE_TUNNELS_PYTHON" - <<'PY'
import json
import os

body = {
    "originalURL": os.environ["PHONE_TUNNELS_PAYLOAD_LONG_URL"],
    "allowDuplicates": False,
}

link_id = os.environ.get("PHONE_TUNNELS_PAYLOAD_LINK_ID", "").strip()
domain = os.environ.get("PHONE_TUNNELS_PAYLOAD_DOMAIN", "").strip()
path = os.environ.get("PHONE_TUNNELS_PAYLOAD_PATH", "").strip()

if link_id:
    if domain:
        body["domain"] = domain
    if path:
        body["path"] = path
else:
    body["domain"] = domain
    if path:
        body["path"] = path

print(json.dumps(body))
PY
}

phone_tunnels_set_shortio_short_url() {
    local api_key="$1"
    local long_url="$2"
    local domain="$3"
    local path="$4"
    local link_id="$5"
    local endpoint="https://api.short.io/links"
    local payload
    local response_path
    local http_status

    if [[ -n "$link_id" ]]; then
        endpoint="$endpoint/$(phone_tunnels_url_encode "$link_id")"
    elif [[ -z "$domain" ]]; then
        phone_tunnels_die "SHORTIO_DOMAIN is required the first time so the script can create the permanent short URL."
    fi

    payload="$(phone_tunnels_build_shortio_payload "$long_url" "$domain" "$path" "$link_id")"
    response_path="$(mktemp)"

    http_status="$(
        curl -sS \
            -o "$response_path" \
            -w "%{http_code}" \
            -X POST \
            -H "authorization: $api_key" \
            -H "accept: application/json" \
            -H "content-type: application/json" \
            --data "$payload" \
            "$endpoint"
    )"

    if [[ "$http_status" -lt 200 || "$http_status" -ge 300 ]]; then
        local response_body
        response_body="$(cat "$response_path" 2>/dev/null || true)"
        rm -f "$response_path"
        phone_tunnels_die "Short.io request failed with HTTP $http_status. $response_body"
    fi

    mapfile -t shortio_result < <(
        PHONE_TUNNELS_SHORTIO_RESPONSE_PATH="$response_path" "$PHONE_TUNNELS_PYTHON" - <<'PY'
import json
import os
import sys

with open(os.environ["PHONE_TUNNELS_SHORTIO_RESPONSE_PATH"], "r", encoding="utf-8") as handle:
    response = json.load(handle)

link_id = response.get("idString") or response.get("id") or ""
short_url = response.get("secureShortURL") or response.get("shortURL") or ""

if not short_url:
    sys.exit(1)

print(str(link_id))
print(short_url)
PY
    ) || {
        rm -f "$response_path"
        phone_tunnels_die "Short.io response did not include a short URL."
    }

    rm -f "$response_path"

    PHONE_TUNNELS_LAST_SHORT_LINK_ID="${shortio_result[0]-}"
    PHONE_TUNNELS_LAST_SHORT_URL="${shortio_result[1]-}"
}
