#!/usr/bin/env bash
# ==============================================================================
# deploy_llamacpp.sh — Automated Setup & Serving Pipeline for llama.cpp
#                     (llama-server) for DocTune Studio
#
# Target: Provides a zero-cost, private OpenAI-compatible endpoint on Port 8080.
#
# Usage:
#   bash scripts/deploy_llamacpp.sh [start|stop|restart|status|logs|download-model]
#
# Default Flags:
#   --port 8080              (Server port, matching DocTune llamacpp preset)
#   --host 0.0.0.0           (Listen on all network interfaces)
#   --ctx 4096               (Context window size)
#   --gpu-layers auto        (Auto-detect: 99 if NVIDIA GPU detected, else 0)
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
    echo "Usage: $0 [start|stop|restart|status|logs|download-model] [options]"
    echo ""
    echo "Options:"
    echo "  --port <port>         Port to bind llama-server (default: 8080)"
    echo "  --host <host>         Host interface (default: 0.0.0.0)"
    echo "  --ctx <size>          KV cache context length (default: 4096)"
    echo "  --model-url <url>     Hugging Face or direct URL to GGUF model"
    echo "  --model-path <path>   Path to existing local .gguf model file"
    echo "  --gpu-layers <N>      Number of layers to offload to GPU (auto: 99 if GPU, 0 if CPU)"
    exit 0
fi

# ── Paths & Defaults ──────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ -n "${LLAMACPP_HOME:-}" ]; then
    LLAMA_DIR="$LLAMACPP_HOME"
elif [ -w "$HOME" ] 2>/dev/null && mkdir -p "$HOME/.doctune/llamacpp" 2>/dev/null; then
    LLAMA_DIR="$HOME/.doctune/llamacpp"
else
    LLAMA_DIR="$REPO_ROOT/.scratch/llamacpp"
fi

# Automatically source repository .env if present
if [ -f "$REPO_ROOT/.env" ]; then
    set -a
    . "$REPO_ROOT/.env" 2>/dev/null || true
    set +a
fi

BIN_DIR="$LLAMA_DIR/bin"
MODEL_DIR="$LLAMA_DIR/models"
LOG_FILE="$LLAMA_DIR/llama-server.log"
PID_FILE="$LLAMA_DIR/llama-server.pid"

DEFAULT_PORT=8080
DEFAULT_HOST="0.0.0.0"
DEFAULT_CTX=4096
# Verified single-file Qwen2.5-1.5B (~1.1 GB): fast on CPU, fits small RAM/disk headroom
DEFAULT_MODEL_NAME="qwen2.5-1.5b-instruct-q4_k_m.gguf"
DEFAULT_MODEL_URL="https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf"

PORT="${PORT:-$DEFAULT_PORT}"
HOST="${HOST:-$DEFAULT_HOST}"
CTX_SIZE="${CTX_SIZE:-$DEFAULT_CTX}"
CUSTOM_MODEL_URL=""
CUSTOM_MODEL_PATH=""
CUSTOM_MODEL_PRESET=""
GPU_LAYERS="auto"

# Parse optional flags
shift || true
while [[ $# -gt 0 ]]; do
    case "$1" in
        --port)
            PORT="$2"; shift 2 ;;
        --host)
            HOST="$2"; shift 2 ;;
        --ctx)
            CTX_SIZE="$2"; shift 2 ;;
        --model-url)
            CUSTOM_MODEL_URL="$2"; shift 2 ;;
        --model-path)
            CUSTOM_MODEL_PATH="$2"; shift 2 ;;
        --model|--preset)
            CUSTOM_MODEL_PRESET="$2"; shift 2 ;;
        --ngl|--gpu-layers)
            GPU_LAYERS="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 [start|stop|restart|status|logs|download-model] [options]"
            echo ""
            echo "Options:"
            echo "  --port <port>         Port to bind llama-server (default: 8080)"
            echo "  --host <host>         Host interface (default: 0.0.0.0)"
            echo "  --ctx <size>          KV cache context length (default: 4096)"
            echo "  --model <preset>      Preset: 1.5b (default, ~1.1GB), 0.5b (~390MB), 7b (~4.7GB)"
            echo "  --model-url <url>     Hugging Face or direct URL to GGUF model"
            echo "  --model-path <path>   Path to existing local .gguf model file"
            echo "  --gpu-layers <N>      Number of layers to offload to GPU (auto: 99 if GPU, 0 if CPU)"
            exit 0
            ;;
        *)
            warn "Unknown parameter: $1"; shift ;;
    esac
done

mkdir -p "$BIN_DIR" "$MODEL_DIR"

# ── Hardware & Environment Probing ────────────────────────────────────────────
detect_hardware() {
    ARCH="$(uname -m)"
    OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
    HAS_CUDA=false

    if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
        HAS_CUDA=true
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 || echo "NVIDIA GPU")
        log "Detected GPU Acceleration: ${BOLD}$GPU_NAME${RESET}"
    else
        info "No NVIDIA GPU detected. Using CPU execution (AVX2/AVX-512/ARM NEON)."
    fi

    if [ "$GPU_LAYERS" = "auto" ]; then
        if [ "$HAS_CUDA" = true ]; then
            GPU_LAYERS=99
        else
            GPU_LAYERS=0
        fi
    fi

    # Ensure libgomp.so.1 is present for OpenMP CPU multithreading
    if [ "$OS" = "linux" ]; then
        HAS_LIBGOMP=false
        if ldconfig -p 2>/dev/null | grep -q "libgomp.so.1"; then
            HAS_LIBGOMP=true
        elif /sbin/ldconfig -p 2>/dev/null | grep -q "libgomp.so.1"; then
            HAS_LIBGOMP=true
        elif [ -f /usr/lib/x86_64-linux-gnu/libgomp.so.1 ] || [ -f /usr/lib/aarch64-linux-gnu/libgomp.so.1 ] || [ -f /lib/x86_64-linux-gnu/libgomp.so.1 ]; then
            HAS_LIBGOMP=true
        fi

        if [ "$HAS_LIBGOMP" = false ]; then
            info "Required OpenMP library (libgomp.so.1) not found. Installing libgomp1..."
            if command -v apt-get &>/dev/null; then
                sudo apt-get update -qq && sudo apt-get install -y -qq libgomp1 || true
            elif command -v dnf &>/dev/null; then
                sudo dnf install -y libgomp || true
            elif command -v yum &>/dev/null; then
                sudo yum install -y libgomp || true
            elif command -v pacman &>/dev/null; then
                sudo pacman -Sy --noconfirm gcc-libs || true
            fi
        fi
    fi
}

# ── Download or Locate llama-server Binary ─────────────────────────────────────
ensure_llama_server() {
    # Check and flatten any nested directory structure from zip extraction (build/bin or bin)
    if [ -d "$BIN_DIR/build/bin" ]; then
        cp -r "$BIN_DIR"/build/bin/* "$BIN_DIR"/
        rm -rf "$BIN_DIR/build"
    elif [ -d "$BIN_DIR/bin" ]; then
        cp -r "$BIN_DIR"/bin/* "$BIN_DIR"/
        rm -rf "$BIN_DIR/bin"
    fi
    chmod +x "$BIN_DIR"/* 2>/dev/null || true

    SERVER_BIN="$BIN_DIR/llama-server"
    if [ -x "$SERVER_BIN" ]; then
        log "Found existing llama-server at: $SERVER_BIN"
        return 0
    fi

    # Check if llama-server is already installed on system PATH
    if command -v llama-server &>/dev/null; then
        SERVER_BIN="$(command -v llama-server)"
        log "Found system llama-server at: $SERVER_BIN"
        return 0
    fi

    info "Fetching latest llama.cpp release metadata from GitHub..."
    detect_hardware

    # Query latest release tag from GitHub API with User-Agent
    LATEST_JSON=$(curl -sSL -H "User-Agent: DocTune-Deploy" "https://api.github.com/repos/ggerganov/llama.cpp/releases/latest" 2>/dev/null || echo "")
    TAG=$(echo "$LATEST_JSON" | grep '"tag_name":' | head -n1 | cut -d '"' -f 4 || echo "b4600")
    if [[ ! "$TAG" =~ ^b[0-9]+$ ]]; then
        info "Notice: GitHub release tag '$TAG' is not in standard 'b<number>' format; pinning to verified release 'b4600'."
        TAG="b4600"
    fi

    info "Target release: ${BOLD}$TAG${RESET} (OS: $OS, Arch: $ARCH, CUDA: $HAS_CUDA)"

    ASSET_NAME=""
    if [ "$OS" = "linux" ]; then
        if [ "$ARCH" = "x86_64" ]; then
            if [ "$HAS_CUDA" = true ]; then
                ASSET_NAME="llama-${TAG}-bin-ubuntu-x64-cuda-cu12.4.zip"
            else
                ASSET_NAME="llama-${TAG}-bin-ubuntu-x64.zip"
            fi
        elif [ "$ARCH" = "aarch64" ]; then
            ASSET_NAME="llama-${TAG}-bin-ubuntu-arm64.zip"
        fi
    elif [ "$OS" = "darwin" ]; then
        ASSET_NAME="llama-${TAG}-bin-macos-arm64.zip"
    fi

    DOWNLOAD_URL="https://github.com/ggerganov/llama.cpp/releases/download/${TAG}/${ASSET_NAME}"
    TMP_ZIP="/tmp/llama-release-$$.zip"

    info "Downloading pre-built release package: $ASSET_NAME..."
    if curl -sSL -f -L "$DOWNLOAD_URL" -o "$TMP_ZIP"; then
        if command -v unzip &>/dev/null; then
            unzip -q -o "$TMP_ZIP" -d "$BIN_DIR"
        elif command -v python3 &>/dev/null; then
            python3 -m zipfile -e "$TMP_ZIP" "$BIN_DIR"
        elif command -v python &>/dev/null; then
            python -m zipfile -e "$TMP_ZIP" "$BIN_DIR"
        elif command -v 7z &>/dev/null; then
            7z x -y "$TMP_ZIP" -o"$BIN_DIR" >/dev/null
        else
            warn "'unzip' command missing. Attempting to install unzip package..."
            (sudo apt-get update -y && sudo apt-get install -y unzip && unzip -q -o "$TMP_ZIP" -d "$BIN_DIR") || true
        fi
        rm -f "$TMP_ZIP"

        # Flatten nested directory structure (llama.cpp packages binaries inside build/bin)
        if [ -d "$BIN_DIR/build/bin" ]; then
            cp -r "$BIN_DIR"/build/bin/* "$BIN_DIR"/
            rm -rf "$BIN_DIR/build"
        elif [ -d "$BIN_DIR/bin" ]; then
            cp -r "$BIN_DIR"/bin/* "$BIN_DIR"/
            rm -rf "$BIN_DIR/bin"
        fi
        chmod +x "$BIN_DIR"/* 2>/dev/null || true
        log "Successfully installed llama-server to $BIN_DIR"
    else
        warn "Direct binary release download failed. Attempting Docker fallback..."
        if command -v docker &>/dev/null; then
            log "Docker is installed. You can run llama.cpp via Docker container."
        else
            err "Could not download llama.cpp binary and Docker is not available. Please install cmake and compile llama.cpp manually."
        fi
    fi
}

# ── Model Download Helper ─────────────────────────────────────────────────────
ensure_model() {
    if [ -n "$CUSTOM_MODEL_PATH" ] && [ -f "$CUSTOM_MODEL_PATH" ]; then
        ACTIVE_MODEL="$CUSTOM_MODEL_PATH"
        log "Using specified local model: $ACTIVE_MODEL"
        return 0
    fi

    # Preset resolutions for verified single-file GGUFs
    if [ "$CUSTOM_MODEL_PRESET" = "7b" ] || [ "$CUSTOM_MODEL_PRESET" = "qwen7b" ]; then
        CUSTOM_MODEL_URL="https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
    elif [ "$CUSTOM_MODEL_PRESET" = "0.5b" ] || [ "$CUSTOM_MODEL_PRESET" = "qwen0.5b" ]; then
        CUSTOM_MODEL_URL="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf"
    elif [ "$CUSTOM_MODEL_PRESET" = "1.5b" ] || [ "$CUSTOM_MODEL_PRESET" = "qwen1.5b" ]; then
        CUSTOM_MODEL_URL="https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf"
    fi

    TARGET_URL="${CUSTOM_MODEL_URL:-$DEFAULT_MODEL_URL}"
    TARGET_NAME="$(basename "$TARGET_URL")"
    TARGET_PATH="$MODEL_DIR/$TARGET_NAME"

    # Strict model validation: verify file exists, size > 100MB, and GGUF magic header
    FILE_SIZE=0
    if [ -f "$TARGET_PATH" ]; then
        FILE_SIZE=$(stat -c%s "$TARGET_PATH" 2>/dev/null || stat -f%z "$TARGET_PATH" 2>/dev/null || echo "0")
    fi

    if [ -f "$TARGET_PATH" ] && [ "$FILE_SIZE" -gt 104857600 ]; then
        MAGIC=$(head -c 4 "$TARGET_PATH" 2>/dev/null || echo "")
        if [ "$MAGIC" = "GGUF" ]; then
            ACTIVE_MODEL="$TARGET_PATH"
            log "Valid GGUF model present: $ACTIVE_MODEL ($(du -h "$ACTIVE_MODEL" | cut -f1))"
            return 0
        else
            warn "Model file has valid size but invalid magic header '$MAGIC'. Purging corrupt file..."
            rm -f "$TARGET_PATH"
        fi
    elif [ -f "$TARGET_PATH" ]; then
        warn "Model file is corrupt or truncated ($(du -h "$TARGET_PATH" | cut -f1) < 100MB). Purging before download..."
        rm -f "$TARGET_PATH"
    fi

    info "Model not found locally or was corrupt. Downloading $TARGET_NAME..."
    info "URL: $TARGET_URL"
    info "Destination: $TARGET_PATH"

    AUTH_HEADER=()
    HF_AUTH="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"
    if [ -n "$HF_AUTH" ]; then
        AUTH_HEADER=(-H "Authorization: Bearer $HF_AUTH")
    fi

    if command -v curl &>/dev/null; then
        curl -f -L --progress-bar "${AUTH_HEADER[@]}" "$TARGET_URL" -o "$TARGET_PATH" || {
            rm -f "$TARGET_PATH"
            err "Failed to download model from $TARGET_URL (HTTP error or invalid link). If model is gated, export HF_TOKEN."
        }
    elif command -v wget &>/dev/null; then
        WGET_HEADER=()
        if [ -n "$HF_AUTH" ]; then
            WGET_HEADER=(--header="Authorization: Bearer $HF_AUTH")
        fi
        wget -q --show-progress "${WGET_HEADER[@]}" "$TARGET_URL" -O "$TARGET_PATH" || {
            rm -f "$TARGET_PATH"
            err "Failed to download model from $TARGET_URL."
        }
    else
        err "Neither curl nor wget found to download model."
    fi

    if [ ! -s "$TARGET_PATH" ]; then
        rm -f "$TARGET_PATH"
        err "Model download failed or file is empty."
    fi

    # Post-download verification
    DOWNLOADED_MAGIC=$(head -c 4 "$TARGET_PATH" 2>/dev/null || echo "")
    if [ "$DOWNLOADED_MAGIC" != "GGUF" ]; then
        rm -f "$TARGET_PATH"
        err "Downloaded file is not a valid GGUF binary (magic header was '$DOWNLOADED_MAGIC'). Check URL or authentication."
    fi

    log "Model download complete and verified: $TARGET_PATH ($(du -h "$TARGET_PATH" | cut -f1))"
    ACTIVE_MODEL="$TARGET_PATH"
}

# ── Process Management Actions ────────────────────────────────────────────────
start_server() {
    detect_hardware

    if [ -f "$PID_FILE" ]; then
        OLD_PID=$(cat "$PID_FILE")
        if ps -p "$OLD_PID" &>/dev/null; then
            warn "llama-server is already running with PID $OLD_PID on port $PORT."
            echo "Run 'bash $0 status' or 'bash $0 stop'."
            exit 0
        else
            rm -f "$PID_FILE"
        fi
    fi

    ensure_llama_server
    ensure_model

    SERVER_BIN="$BIN_DIR/llama-server"
    if [ ! -x "$SERVER_BIN" ] && [ -x "$BIN_DIR/build/bin/llama-server" ]; then
        cp -r "$BIN_DIR"/build/bin/* "$BIN_DIR"/
        rm -rf "$BIN_DIR/build"
        chmod +x "$BIN_DIR"/* 2>/dev/null || true
        SERVER_BIN="$BIN_DIR/llama-server"
    elif [ ! -x "$SERVER_BIN" ] && [ -x "$BIN_DIR/bin/llama-server" ]; then
        cp -r "$BIN_DIR"/bin/* "$BIN_DIR"/
        rm -rf "$BIN_DIR/bin"
        chmod +x "$BIN_DIR"/* 2>/dev/null || true
        SERVER_BIN="$BIN_DIR/llama-server"
    elif [ ! -x "$SERVER_BIN" ] && command -v llama-server &>/dev/null; then
        SERVER_BIN="$(command -v llama-server)"
    fi
    if [ ! -x "$SERVER_BIN" ]; then
        err "llama-server binary not found or not executable at: $SERVER_BIN. Ensure pre-built binary downloaded or compile locally."
    fi

    # Export dynamic library path so shared libs (libllama.so, libggml.so, etc.) resolve cleanly
    export LD_LIBRARY_PATH="$BIN_DIR:${LD_LIBRARY_PATH:-}"

    info "Starting llama-server..."
    info "  - Model       : $ACTIVE_MODEL"
    info "  - Host:Port   : $HOST:$PORT"
    info "  - Context Size: $CTX_SIZE tokens"
    info "  - GPU Layers  : $GPU_LAYERS"

    API_KEY_ARG=()
    if [ -n "${LLAMACPP_API_KEY:-}" ] && [ "$LLAMACPP_API_KEY" != "not-needed" ]; then
        API_KEY_ARG=(--api-key "$LLAMACPP_API_KEY")
    fi

    nohup "$SERVER_BIN" \
        -m "$ACTIVE_MODEL" \
        --host "$HOST" \
        --port "$PORT" \
        -c "$CTX_SIZE" \
        -ngl "$GPU_LAYERS" \
        "${API_KEY_ARG[@]}" \
        > "$LOG_FILE" 2>&1 &

    PID=$!
    echo "$PID" > "$PID_FILE"

    log "llama-server started with PID $PID."
    info "Waiting for endpoint readiness at http://127.0.0.1:$PORT/v1/models..."

    READY=false
    for i in {1..20}; do
        sleep 1
        if curl -s "http://127.0.0.1:$PORT/v1/models" &>/dev/null; then
            READY=true
            break
        fi
        if ! ps -p "$PID" &>/dev/null; then
            err "Server process terminated unexpectedly. Check logs: cat $LOG_FILE"
        fi
    done

    if [ "$READY" = true ]; then
        echo ""
        echo -e "${GREEN}${BOLD}══════════════════════════════════════════════════════════════${RESET}"
        echo -e "${GREEN}${BOLD}   llama.cpp (llama-server) IS READY & SERVING!${RESET}"
        echo -e "${GREEN}${BOLD}══════════════════════════════════════════════════════════════${RESET}"
        echo -e "   API Endpoint   : ${CYAN}http://localhost:$PORT/v1${RESET}"
        echo -e "   Docker URL     : ${CYAN}http://host.docker.internal:$PORT/v1${RESET}"
        echo -e "   Active Model   : $(basename "$ACTIVE_MODEL")"
        echo -e "   DocTune Preset : Select 'llama.cpp Server' in DocTune Studio"
        echo -e "   Log File       : $LOG_FILE"
        echo -e "${GREEN}${BOLD}══════════════════════════════════════════════════════════════${RESET}"
    else
        warn "Server started but did not respond to /v1/models within 20s. Check logs: $LOG_FILE"
    fi
}

stop_server() {
    if [ ! -f "$PID_FILE" ]; then
        info "No running llama-server PID file found."
        return 0
    fi

    PID=$(cat "$PID_FILE")
    if ps -p "$PID" &>/dev/null; then
        info "Stopping llama-server (PID $PID)..."
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
        log "llama-server stopped."
    else
        info "Process $PID is not running."
    fi
    rm -f "$PID_FILE"
}

status_server() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p "$PID" &>/dev/null; then
            log "llama-server is ${GREEN}ACTIVE${RESET} (PID: $PID)"
            ps -u -p "$PID" | tail -n 1
            echo ""
            info "Probing API health at http://127.0.0.1:$PORT/v1/models..."
            HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/v1/models" || echo "000")
            if [ "$HTTP_CODE" = "200" ]; then
                log "Endpoint HTTP 200 OK — Ready for DocTune queries!"
                curl -s "http://127.0.0.1:$PORT/v1/models" | head -c 200
                echo ""
            else
                warn "Endpoint returned HTTP $HTTP_CODE"
            fi
            return 0
        fi
    fi
    info "llama-server is ${RED}NOT RUNNING${RESET}."
}

show_logs() {
    if [ -f "$LOG_FILE" ]; then
        tail -n 50 -f "$LOG_FILE"
    else
        info "No log file found at $LOG_FILE."
    fi
}

# ── Main Entrypoint Dispatcher ────────────────────────────────────────────────
case "$ACTION" in
    start)
        start_server
        ;;
    stop)
        stop_server
        ;;
    restart)
        stop_server
        sleep 1
        start_server
        ;;
    status)
        status_server
        ;;
    logs)
        show_logs
        ;;
    download-model)
        ensure_model
        ;;
    *)
        err "Unknown action '$ACTION'. Supported actions: start, stop, restart, status, logs, download-model."
        ;;
esac
