#!/usr/bin/env bash
# ==============================================================================
# audit_deploy.sh — Comprehensive Pre-Flight System & Deployment Audit
#                   for DocTune Studio
#
# Target: Inspects physical/virtual hardware, GPU/CPU acceleration readiness,
#         Docker daemon status, port conflicts, and validates the default
#         llama.cpp + Qwen 2.5 1.5B GGUF local inference engine.
#
# Usage:
#   bash scripts/audit_deploy.sh
#   bash scripts/deploy.sh audit
# ==============================================================================
set -euo pipefail

# ── Styling & Color Palette ───────────────────────────────────────────────────
GREEN="\033[0;32m"; YELLOW="\033[1;33m"; RED="\033[0;31m"
CYAN="\033[0;36m"; BOLD="\033[1m"; DIM="\033[2m"; RESET="\033[0m"

pass() { echo -e "  [${GREEN}PASS${RESET}] $1"; }
warn() { echo -e "  [${YELLOW}WARN${RESET}] $1"; }
fail() { echo -e "  [${RED}FAIL${RESET}] $1"; }
info() { echo -e "  [${CYAN}INFO${RESET}] $1"; }

WARNINGS=0
ERRORS=0

echo -e "\n${BOLD}${CYAN}══════════════════════════════════════════════════════════════════${RESET}"
echo -e "${BOLD}${CYAN}   DocTune Studio — Pre-Flight System & Deployment Audit        ${RESET}"
echo -e "${BOLD}${CYAN}══════════════════════════════════════════════════════════════════${RESET}\n"

# ── 1. Host & Operating System ────────────────────────────────────────────────
echo -e "${BOLD}1. Host & Operating System${RESET}"
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS_NAME="${PRETTY_NAME:-$NAME}"
else
    OS_NAME="$(uname -s)"
fi
KERNEL="$(uname -r)"
ARCH="$(uname -m)"

info "OS Distribution : $OS_NAME"
info "Kernel Version  : $KERNEL"
info "Architecture    : $ARCH"

if [ "$ARCH" != "x86_64" ] && [ "$ARCH" != "aarch64" ]; then
    warn "Unsupported architecture '$ARCH'. x86_64 or aarch64 (ARM64) is strongly recommended."
    WARNINGS=$((WARNINGS + 1))
else
    pass "Architecture '$ARCH' is fully supported."
fi

# ── 2. CPU, Memory & Swap Capacity ───────────────────────────────────────────
echo -e "\n${BOLD}2. Compute & Memory Headroom${RESET}"
CPU_CORES=$(nproc 2>/dev/null || echo "1")
CPU_MODEL=$(grep -m1 "model name" /proc/cpuinfo 2>/dev/null | sed -E "s/^model name\s*:\s*//" | tr -s " " || echo "Generic CPU")
[ -z "$CPU_MODEL" ] && CPU_MODEL="Generic CPU"

TOTAL_MEM_MB=$(free -m 2>/dev/null | awk '/Mem:/ {print $2}' || echo "0")
FREE_MEM_MB=$(free -m 2>/dev/null | awk '/Mem:/ {print ($7 != "" ? $7 : $4)}' || echo "0")
TOTAL_MEM_GB=$(awk -v m="$TOTAL_MEM_MB" 'BEGIN {printf "%.2f", m/1024}')
FREE_MEM_GB=$(awk -v m="$FREE_MEM_MB" 'BEGIN {printf "%.2f", m/1024}')

SWAP_TOTAL_MB=$(free -m 2>/dev/null | awk '/Swap:/ {print $2}' || echo "0")

info "CPU Model       : $CPU_MODEL"
info "CPU Cores       : $CPU_CORES vCPU(s)"
info "Physical Memory : ${FREE_MEM_GB} GB free of ${TOTAL_MEM_GB} GB total"
info "Swap Space      : ${SWAP_TOTAL_MB} MB"

if [ "$CPU_CORES" -lt 2 ]; then
    warn "System has only $CPU_CORES vCPU. Docker builds and model inference will be slow."
    WARNINGS=$((WARNINGS + 1))
else
    pass "CPU core count ($CPU_CORES cores) meets operational requirements."
fi

if [ "$TOTAL_MEM_MB" -lt 3800 ]; then
    fail "RAM is under 4GB (${TOTAL_MEM_GB} GB). Platform requires at least 4GB (8GB recommended)."
    ERRORS=$((ERRORS + 1))
elif [ "$TOTAL_MEM_MB" -lt 7500 ]; then
    warn "RAM is constrained (${TOTAL_MEM_GB} GB). For full QLoRA fine-tuning, 16GB+ or a dedicated GPU VM is recommended."
    WARNINGS=$((WARNINGS + 1))
else
    pass "Memory capacity (${TOTAL_MEM_GB} GB) is adequate for container execution."
fi

if [ "$SWAP_TOTAL_MB" -lt 1024 ]; then
    warn "Swap space is 0MB or < 1GB. During heavy builds, an active swapfile prevents kernel OOM crashes."
    echo -e "         ${DIM}Quick fix: sudo fallocate -l 4G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile${RESET}"
    WARNINGS=$((WARNINGS + 1))
else
    pass "Swap configuration (${SWAP_TOTAL_MB} MB) provides memory overflow headroom."
fi

# ── 3. Storage & Disk Partition Headroom ──────────────────────────────────────
echo -e "\n${BOLD}3. Storage & Disk Partition Headroom${RESET}"
DISK_FREE_MB=$(df -P -m . 2>/dev/null | awk 'NR==2 {print $4}' || echo "0")
DISK_TOTAL_GB=$(df -P -h . 2>/dev/null | awk 'NR==2 {print $2}' || echo "0")
DISK_FREE_GB=$(df -P -h . 2>/dev/null | awk 'NR==2 {print $4}' || echo "0")

info "Filesystem Total: $DISK_TOTAL_GB"
info "Available Disk  : $DISK_FREE_GB free (${DISK_FREE_MB} MB)"

if [ "$DISK_FREE_MB" -lt 15360 ]; then
    fail "Available disk is critical (< 15 GB). Docker layer extraction and ML models will fail."
    echo -e "         ${DIM}On GCP: resize your boot disk via 'gcloud compute disks resize <DISK> --size=100GB'${RESET}"
    ERRORS=$((ERRORS + 1))
elif [ "$DISK_FREE_MB" -lt 25600 ]; then
    warn "Available disk is moderate ($DISK_FREE_GB free < 25 GB). Pruning unused Docker images before build is advised."
    WARNINGS=$((WARNINGS + 1))
else
    pass "Available disk headroom ($DISK_FREE_GB free) is sufficient for container builds."
fi

# ── 4. Hardware Acceleration & PyTorch Target Device ─────────────────────────
echo -e "\n${BOLD}4. Hardware Acceleration & PyTorch Target Device${RESET}"
HAS_NVIDIA_GPU=false
DETECTED_TARGET="cpu"

if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    HAS_NVIDIA_GPU=true
    DETECTED_TARGET="cuda"
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 || echo "NVIDIA GPU")
    DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1 || echo "Unknown")
    VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n1 || echo "0")
    VRAM_GB=$(awk -v v="$VRAM_MB" 'BEGIN {printf "%.1f", v/1024}')
    
    pass "NVIDIA GPU Detected : $GPU_NAME ($VRAM_GB GB VRAM, Driver $DRIVER_VER)"
    pass "Recommended Target  : ${BOLD}TARGET_DEVICE=cuda${RESET} (Full QLoRA, 4-bit bitsandbytes & FlashAttention)"
else
    info "GPU Acceleration    : No NVIDIA GPU detected (Standard CPU instance)"
    pass "Recommended Target  : ${BOLD}TARGET_DEVICE=cpu${RESET} (Ultra-lightweight wheels; saves ~10GB bloat & ~30 min build time)"
    info "Execution Mode      : Native CPU vectorization (AVX2/AVX-512) via llama.cpp"
fi

# ── 5. Docker Engine & Compose Daemon ─────────────────────────────────────────
echo -e "\n${BOLD}5. Docker Engine & Container Runtime${RESET}"
if command -v docker &>/dev/null; then
    DOCKER_VER=$(docker --version 2>/dev/null || echo "Unknown")
    pass "Docker Installed    : $DOCKER_VER"
    
    if docker info &>/dev/null; then
        pass "Docker Daemon Status: Active & responding without root escalation"
    else
        warn "Docker daemon is not responding or current user lacks group access."
        echo -e "         ${DIM}Quick fix: sudo usermod -aG docker $USER && newgrp docker${RESET}"
        WARNINGS=$((WARNINGS + 1))
    fi
else
    fail "Docker is not installed on this system."
    echo -e "         ${DIM}Install via: curl -fsSL https://get.docker.com | sh${RESET}"
    ERRORS=$((ERRORS + 1))
fi

if docker compose version &>/dev/null; then
    COMPOSE_VER=$(docker compose version 2>/dev/null || echo "")
    pass "Docker Compose      : $COMPOSE_VER (V2 Plugin)"
elif command -v docker-compose &>/dev/null; then
    COMPOSE_VER=$(docker-compose --version 2>/dev/null || echo "")
    warn "Docker Compose      : $COMPOSE_VER (Legacy standalone binary)"
else
    fail "Docker Compose is missing."
    ERRORS=$((ERRORS + 1))
fi

# ── 6. Port Availability & Collision Scan ─────────────────────────────────────
echo -e "\n${BOLD}6. Port Allocation & Collision Scan${RESET}"
check_port() {
    local port=$1
    local service=$2
    local is_optional=${3:-false}
    
    local in_use=false
    local proc=""
    if command -v ss &>/dev/null; then
        proc=$(ss -tulpn "sport = :$port" 2>/dev/null | grep -v "Netid" | head -n1 || true)
        [ -n "$proc" ] && in_use=true
    elif command -v netstat &>/dev/null; then
        proc=$(netstat -tulpn 2>/dev/null | grep ":$port " | head -n1 || true)
        [ -n "$proc" ] && in_use=true
    elif command -v lsof &>/dev/null; then
        proc=$(lsof -i :"$port" -sTCP:LISTEN 2>/dev/null | grep -v "COMMAND" | head -n1 || true)
        [ -n "$proc" ] && in_use=true
    fi

    if [ "$in_use" = true ]; then
        if [ "$port" = "8080" ]; then
            info "Port $port ($service) is currently bound (expected if host llama-server is active)."
        else
            warn "Port $port ($service) is currently in use by an existing process."
            WARNINGS=$((WARNINGS + 1))
        fi
    else
        pass "Port $port ($service) is free and available."
    fi
}

check_port 80    "DocTune Web Studio UI"
check_port 5432  "PostgreSQL Relational DB"
check_port 6379  "Redis Broker & Pub/Sub"
check_port 9000  "MinIO S3 API Endpoint"
check_port 9001  "MinIO S3 Web Console"
check_port 8888  "Dozzle Container Logs"
check_port 8080  "llama.cpp Inference Gateway" true

# ── 7. Default Model Audit: llama.cpp & Qwen 2.5 1.5B GGUF ─────────────────────
echo -e "\n${BOLD}7. Default Model Audit: llama.cpp + Qwen 2.5 1.5B GGUF${RESET}"

# Resolve LLAMA directory consistent with deploy_llamacpp.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ -n "${LLAMACPP_HOME:-}" ]; then
    LLAMA_DIR="$LLAMACPP_HOME"
elif [ -w "$HOME" ] 2>/dev/null && [ -d "$HOME/.doctune/llamacpp" ]; then
    LLAMA_DIR="$HOME/.doctune/llamacpp"
elif [ -d "$REPO_ROOT/.scratch/llamacpp" ]; then
    LLAMA_DIR="$REPO_ROOT/.scratch/llamacpp"
else
    LLAMA_DIR="$HOME/.doctune/llamacpp"
fi

SERVER_BIN=""
if [ -x "$LLAMA_DIR/bin/llama-server" ]; then
    SERVER_BIN="$LLAMA_DIR/bin/llama-server"
elif command -v llama-server &>/dev/null; then
    SERVER_BIN="$(command -v llama-server)"
fi

if [ -n "$SERVER_BIN" ]; then
    pass "llama-server Binary : Installed at $SERVER_BIN"
else
    info "llama-server Binary : Not yet downloaded (will auto-download on first deployment run)"
fi

DEFAULT_MODEL="qwen2.5-1.5b-instruct-q4_k_m.gguf"
MODEL_PATH="$LLAMA_DIR/models/$DEFAULT_MODEL"

if [ -f "$MODEL_PATH" ]; then
    FILE_SIZE=$(stat -c%s "$MODEL_PATH" 2>/dev/null || stat -f%z "$MODEL_PATH" 2>/dev/null || echo "0")
    if [ "$FILE_SIZE" -gt 104857600 ]; then
        MAGIC=$(head -c 4 "$MODEL_PATH" 2>/dev/null || echo "")
        if [ "$MAGIC" = "GGUF" ]; then
            SIZE_HR=$(du -h "$MODEL_PATH" 2>/dev/null | cut -f1)
            pass "Default Model File  : Valid $DEFAULT_MODEL ($SIZE_HR, verified GGUF header)"
        else
            warn "Default Model File  : File exists but magic header is invalid. Model will be re-downloaded."
            WARNINGS=$((WARNINGS + 1))
        fi
    else
        warn "Default Model File  : File is truncated (< 100MB). Model will be re-downloaded."
        WARNINGS=$((WARNINGS + 1))
    fi
else
    info "Default Model File  : $DEFAULT_MODEL not yet cached (~1.1 GB will download on start)"
fi

# Check live endpoint if active
if curl -s -m 2 "http://127.0.0.1:8080/health" &>/dev/null || curl -s -m 2 "http://127.0.0.1:8080/v1/models" &>/dev/null; then
    pass "Active Server Check : llama-server is LIVE and responding at http://127.0.0.1:8080/v1"
else
    info "Active Server Check : llama-server is currently standby (starts via 'bash scripts/deploy.sh --with-llamacpp')"
fi

# ── 8. Summary & Prescriptive Action Plan ─────────────────────────────────────
echo -e "\n${BOLD}${CYAN}══════════════════════════════════════════════════════════════════${RESET}"
echo -e "${BOLD}   Audit Results Summary & Recommendations                      ${RESET}"
echo -e "${BOLD}${CYAN}══════════════════════════════════════════════════════════════════${RESET}"

if [ "$ERRORS" -eq 0 ] && [ "$WARNINGS" -eq 0 ]; then
    echo -e "   Status : ${GREEN}${BOLD}OPTIMAL READINESS (Score: 100/100)${RESET}"
    echo -e "   Target : ${BOLD}TARGET_DEVICE=$DETECTED_TARGET${RESET}"
elif [ "$ERRORS" -eq 0 ]; then
    echo -e "   Status : ${YELLOW}${BOLD}READY WITH ADVISORIES (${WARNINGS} warnings)${RESET}"
    echo -e "   Target : ${BOLD}TARGET_DEVICE=$DETECTED_TARGET${RESET}"
else
    echo -e "   Status : ${RED}${BOLD}ACTION REQUIRED BEFORE DEPLOYMENT (${ERRORS} blocking errors)${RESET}"
fi

echo ""
echo -e "   ${BOLD}Recommended Next Steps:${RESET}"
if [ "$DETECTED_TARGET" = "cpu" ]; then
    echo -e "   1. Deploy stack with lightweight CPU PyTorch and local llama.cpp:"
    echo -e "      ${CYAN}bash scripts/deploy.sh --with-llamacpp${RESET}"
    echo -e "      ${DIM}(Automatically applies TARGET_DEVICE=cpu; saves ~10GB and avoids build crashes)${RESET}"
else
    echo -e "   1. Deploy stack with full GPU CUDA acceleration and llama.cpp:"
    echo -e "      ${CYAN}bash scripts/deploy.sh --with-llamacpp${RESET}"
    echo -e "      ${DIM}(Automatically activates CUDA PyTorch and GPU offloading)${RESET}"
fi

if [ "$WARNINGS" -gt 0 ] && [ "$SWAP_TOTAL_MB" -lt 1024 ]; then
    echo -e "   2. Provision a 4GB swapfile if on a constrained VM:"
    echo -e "      ${CYAN}sudo fallocate -l 4G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile${RESET}"
fi

echo -e "${BOLD}${CYAN}══════════════════════════════════════════════════════════════════${RESET}\n"

exit "$ERRORS"
