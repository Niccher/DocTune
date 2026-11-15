#!/usr/bin/env bash
# ==============================================================================
# deploy_ollama.sh — Automated Setup & Serving Pipeline for Ollama
#                    for DocTune Studio (Universal Teacher LLM Gateway)
#
# Target: Provides an air-gapped, zero-cost OpenAI-compatible endpoint on Port 11434.
#
# Usage:
#   bash scripts/deploy_ollama.sh [start|stop|restart|status|logs|pull-model] [options]
#
# Options:
#   --port <port>         Port to bind Ollama (default: 11434)
#   --host <host>         Host interface (default: 0.0.0.0)
#   --model <name>        Model to auto-pull (default: qwen2.5:1.5b)
# ==============================================================================
set -euo pipefail

# ── Styling & Colors ──────────────────────────────────────────────────────────
GREEN="\033[0;32m"; YELLOW="\033[1;33m"; RED="\033[0;31m"
CYAN="\033[0;36m"; BOLD="\033[1m"; RESET="\033[0m"

log()   { echo -e "${GREEN}[OK]${RESET}   $1"; }
warn()  { echo -e "${YELLOW}[WARN]${RESET} $1"; }
info()  { echo -e "${CYAN}[INFO]${RESET} $1"; }
err()   { echo -e "${RED}[FAIL]${RESET} $1"; exit 1; }

# ── CLI Arguments & Early Help ─────────────────────────────────────────────────
ACTION="${1:-start}"
if [ "$ACTION" = "--help" ] || [ "$ACTION" = "-h" ]; then
    echo "Usage: $0 [start|stop|restart|status|logs|pull-model] [options]"
    echo ""
    echo "Options:"
    echo "  --port <port>         Port to bind Ollama (default: 11434)"
    echo "  --host <host>         Host interface (default: 0.0.0.0)"
    echo "  --model <name>        Model to pull (default: qwen2.5:1.5b)"
    exit 0
fi

# ── Paths & Defaults ──────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ -n "${OLLAMA_HOME:-}" ]; then
    OLLAMA_DIR="$OLLAMA_HOME"
elif [ -w "$HOME" ] 2>/dev/null && mkdir -p "$HOME/.doctune/ollama" 2>/dev/null; then
    OLLAMA_DIR="$HOME/.doctune/ollama"
else
    OLLAMA_DIR="$REPO_ROOT/.scratch/ollama"
fi

# Automatically source repository .env if present
if [ -f "$REPO_ROOT/.env" ]; then
    set -a
    . "$REPO_ROOT/.env" 2>/dev/null || true
    set +a
fi

LOG_FILE="$OLLAMA_DIR/ollama.log"
PID_FILE="$OLLAMA_DIR/ollama.pid"

DEFAULT_PORT=11434
DEFAULT_HOST="0.0.0.0"
# Verified lightweight default: Qwen 2.5 1.5B fits small CPU VMs effortlessly
DEFAULT_MODEL="qwen2.5:1.5b"

PORT="${PORT:-$DEFAULT_PORT}"
HOST="${HOST:-$DEFAULT_HOST}"
TARGET_MODEL="$DEFAULT_MODEL"

# Parse optional flags
shift || true
while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)
            PORT="$2"; shift 2 ;;
        --host)
            HOST="$2"; shift 2 ;;
        --model)
            TARGET_MODEL="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 [start|stop|restart|status|logs|pull-model] [options]"
            exit 0
            ;;
        *)
            warn "Unknown parameter: $1"; shift ;;
    esac
done

mkdir -p "$OLLAMA_DIR"

# ── Ensure Ollama CLI Binary is Installed ─────────────────────────────────────
ensure_ollama() {
    if command -v ollama &>/dev/null; then
        log "Found existing Ollama binary at: $(command -v ollama)"
        return 0
    fi

    # Check common non-standard PATH locations
    for loc in /usr/local/bin/ollama /usr/bin/ollama "$HOME/.local/bin/ollama" "$HOME/bin/ollama"; do
        if [ -x "$loc" ]; then
            log "Found Ollama binary at: $loc"
            export PATH="$(dirname "$loc"):$PATH"
            return 0
        fi
    done

    info "Ollama binary not found on host. Installing official Ollama release..."
    if command -v curl &>/dev/null; then
        curl -fsSL https://ollama.com/install.sh | sh || {
            warn "Standard curl install failed; attempting manual binary download..."
            TMP_TGZ="/tmp/ollama-linux.tgz"
            curl -fsSL "https://ollama.com/download/ollama-linux-amd64.tgz" -o "$TMP_TGZ"
            sudo tar -C /usr/local -xzf "$TMP_TGZ" || tar -C "$HOME/.local" -xzf "$TMP_TGZ"
            rm -f "$TMP_TGZ"
        }
    else
        err "curl is required to install Ollama. Run: sudo apt-get install -y curl"
    fi

    if ! command -v ollama &>/dev/null; then
        err "Failed to install Ollama automatically. Please install manually: https://ollama.com/download"
    fi

    log "Ollama successfully installed: $(ollama --version 2>/dev/null || echo 'OK')"
}

# ── Ensure Default Model is Pulled ───────────────────────────────────────────
ensure_model() {
    export OLLAMA_HOST="$HOST:$PORT"
    info "Checking if model '${TARGET_MODEL}' is downloaded in Ollama..."
    
    if ollama list 2>/dev/null | grep -q "${TARGET_MODEL}"; then
        log "Model '${TARGET_MODEL}' is already cached in Ollama"
        return 0
    fi

    info "Pulling model '${TARGET_MODEL}' (this may take 1-2 minutes on first run)..."
    if ollama pull "${TARGET_MODEL}"; then
        log "Successfully pulled '${TARGET_MODEL}' into Ollama"
    else
        warn "Failed pulling '${TARGET_MODEL}'. You can pull it later via: ollama pull ${TARGET_MODEL}"
    fi
}

# ── Server Actions ────────────────────────────────────────────────────────────
start_server() {
    # If already running on host port, verify and reuse
    if curl -s "http://127.0.0.1:$PORT/api/tags" &>/dev/null; then
        log "Ollama is already running and responding on http://127.0.0.1:$PORT"
        ensure_model
        print_banner
        return 0
    fi

    # Check if systemd manages an ollama service
    if command -v systemctl &>/dev/null && systemctl is-active --quiet ollama 2>/dev/null; then
        log "Ollama systemd service is active."
        ensure_model
        print_banner
        return 0
    fi

    ensure_ollama

    export OLLAMA_HOST="$HOST:$PORT"
    export OLLAMA_MODELS="${OLLAMA_MODELS:-$OLLAMA_DIR/models}"
    mkdir -p "$OLLAMA_MODELS"

    info "Starting Ollama server daemon on $HOST:$PORT..."
    nohup ollama serve > "$LOG_FILE" 2>&1 &
    PID=$!
    echo "$PID" > "$PID_FILE"

    log "Ollama started in background with PID $PID."
    info "Waiting for endpoint readiness at http://127.0.0.1:$PORT/api/tags..."

    READY=false
    for i in {1..25}; do
        sleep 1
        if curl -s "http://127.0.0.1:$PORT/api/tags" &>/dev/null; then
            READY=true
            break
        fi
        if ! ps -p "$PID" &>/dev/null; then
            err "Ollama process terminated unexpectedly. Check logs: cat $LOG_FILE"
        fi
    done

    if [ "$READY" = true ]; then
        ensure_model
        print_banner
    else
        warn "Ollama server started but did not respond to /api/tags within 25s. Check logs: $LOG_FILE"
    fi
}

print_banner() {
    echo ""
    echo -e "${GREEN}${BOLD}══════════════════════════════════════════════════════════════${RESET}"
    echo -e "${GREEN}${BOLD}   OLLAMA SERVER IS READY & SERVING!${RESET}"
    echo -e "${GREEN}${BOLD}══════════════════════════════════════════════════════════════${RESET}"
    echo -e "   API Endpoint   : ${CYAN}http://localhost:$PORT/v1${RESET}"
    echo -e "   Docker URL     : ${CYAN}http://host.docker.internal:$PORT/v1${RESET}"
    echo -e "   Active Model   : ${BOLD}$TARGET_MODEL${RESET}"
    echo -e "   DocTune Preset : Select 'Ollama (Local / Free & Private)' in Studio"
    echo -e "   Log File       : $LOG_FILE"
    echo -e "${GREEN}${BOLD}══════════════════════════════════════════════════════════════${RESET}"
}

stop_server() {
    # Check systemd first
    if command -v systemctl &>/dev/null && systemctl is-active --quiet ollama 2>/dev/null; then
        info "Stopping Ollama systemd service..."
        sudo systemctl stop ollama 2>/dev/null || true
    fi

    if [ ! -f "$PID_FILE" ]; then
        # Check if process is running by name
        if pgrep -x "ollama" &>/dev/null; then
            info "Stopping Ollama background process..."
            pkill -x "ollama" 2>/dev/null || true
            log "Ollama stopped."
        else
            info "No running Ollama PID file or process found."
        fi
        return 0
    fi

    PID=$(cat "$PID_FILE")
    if ps -p "$PID" &>/dev/null; then
        info "Stopping Ollama (PID $PID)..."
        kill "$PID" 2>/dev/null || true
        for i in {1..10}; do
            if ! ps -p "$PID" &>/dev/null; then
                break
            fi
            sleep 0.5
        done
        if ps -p "$PID" &>/dev/null; then
            warn "Process did not terminate gracefully; sending SIGKILL..."
            kill -9 "$PID" 2>/dev/null || true
        fi
        log "Ollama stopped."
    else
        info "Process $PID is not running."
    fi
    rm -f "$PID_FILE"
}

status_server() {
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/api/tags" 2>/dev/null || echo "000")
    if [ "$HTTP_CODE" = "200" ]; then
        log "Ollama is ${GREEN}ACTIVE & SERVING${RESET} on http://127.0.0.1:$PORT"
        echo ""
        echo -e "   ${BOLD}Available Models:${RESET}"
        ollama list 2>/dev/null | sed 's/^/   /' || true
        echo ""
        return 0
    fi

    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p "$PID" &>/dev/null; then
            warn "Ollama process PID $PID is alive but port $PORT is not answering (HTTP $HTTP_CODE)."
            return 0
        fi
    fi

    info "Ollama server is ${RED}NOT RUNNING${RESET}."
}

show_logs() {
    if [ -f "$LOG_FILE" ]; then
        tail -n 50 -f "$LOG_FILE"
    else
        info "No log file found at $LOG_FILE."
    fi
}

# ── Main Dispatcher ───────────────────────────────────────────────────────────
case "$ACTION" in
    start)
        start_server ;;
    stop)
        stop_server ;;
    restart)
        stop_server
        sleep 1
        start_server
        ;;
    status)
        status_server ;;
    logs)
        show_logs ;;
    pull-model|pull)
        export OLLAMA_HOST="$HOST:$PORT"
        ensure_model ;;
    *)
        echo "Usage: $0 [start|stop|restart|status|logs|pull-model] [options]"
        exit 1
        ;;
esac
