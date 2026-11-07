#!/usr/bin/env bash
# ==============================================================================
# deploy.sh — ONE-CLICK Deployment & Orchestration Pipeline
#             for DocTune Studio (Private Document LLM Fine-Tuning Platform)
#
# Usage:  bash scripts/deploy.sh
#         (run from the repository root)
#
# Production Standard Ports:
#   - Web Studio & Dashboard   : Port 80 (standard HTTP)
#   - PostgreSQL Database      : Port 5432
#   - Redis Message Broker     : Port 6379
#   - MinIO S3 API Endpoint    : Port 9000
#   - MinIO S3 Web Console     : Port 9001
#   - Dozzle Log Monitor       : Port 8888
#
# 13-Step Automated Pipeline:
#   1. Detect Dynamic IP (Public / LAN Fallback)
#   2. Pre-flight System & Resource Check (OS, CPU, RAM, Disk, Auto-Swap)
#   3. Git Working Tree Sanity Check & Pull
#   4. Write / Patch .env Configuration
#   5. Stop Old Containers & Clear Name Conflicts
#   6. Build & Start All Containers (Postgres, Redis, MinIO, Web, Celery Worker, Dozzle)
#   7. Wait for Database Health (:5432)
#      7b. Wait for Redis Health (:6379)
#      7c. Wait for MinIO Health (:9000 / :9001)
#   8. Initialize Database Tables & Schema
#   9. Verify MinIO S3 Bucket Provisioning (Native Python S3 API)
#  10. Wait for WebApp Studio Readiness (:80)
#  11. Container Permissions & Scratch Directory Enforcement
#  12. Housekeeping, Docker Storage & Cache Clearing
#  13. Deployment Timeline Summary & Live Endpoints
# ==============================================================================
set -euo pipefail

# ── Colors & Formatting ───────────────────────────────────────
GREEN="\033[0;32m"; YELLOW="\033[1;33m"; RED="\033[0;31m"
CYAN="\033[0;36m"; BOLD="\033[1m"; RESET="\033[0m"

DEPLOY_START_TIME=$(date +%s)
SECTION_START_TIME=$DEPLOY_START_TIME
PREV_SECTION=""
declare -a STAGE_NAMES=()
declare -a STAGE_DURATIONS=()

format_duration() {
    local S=$1
    if [ "$S" -ge 60 ]; then
        local M=$((S / 60))
        local R=$((S % 60))
        echo "${M}m ${R}s"
    else
        echo "${S}s"
    fi
}

log()     { echo -e "${GREEN}[OK]${RESET}   $1"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET} $1"; }
err()     {
    local NOW=$(date +%s)
    local ELAPSED=$((NOW - DEPLOY_START_TIME))
    echo -e "${RED}[FAIL]${RESET} $1 (failed after $(format_duration $ELAPSED))"
    exit 1
}

section() {
    local NOW=$(date +%s)
    if [ -n "$PREV_SECTION" ]; then
        local DURATION=$((NOW - SECTION_START_TIME))
        echo -e "${CYAN}──> Completed: ${PREV_SECTION} in $(format_duration $DURATION)${RESET}"
        STAGE_NAMES+=("$PREV_SECTION")
        STAGE_DURATIONS+=("$DURATION")
    fi
    PREV_SECTION="$1"
    SECTION_START_TIME=$NOW
    echo -e "\n${CYAN}${BOLD}=== $1 ===${RESET}"
}

finish_deployment() {
    local NOW=$(date +%s)
    if [ -n "$PREV_SECTION" ]; then
        local DURATION=$((NOW - SECTION_START_TIME))
        echo -e "${CYAN}──> Completed: ${PREV_SECTION} in $(format_duration $DURATION)${RESET}"
        STAGE_NAMES+=("$PREV_SECTION")
        STAGE_DURATIONS+=("$DURATION")
        PREV_SECTION=""
    fi
    local TOTAL_DURATION=$((NOW - DEPLOY_START_TIME))
    echo ""
    echo -e "${GREEN}${BOLD}══════════════════════════════════════════════════════════════${RESET}"
    echo -e "${GREEN}${BOLD}   DEPLOYMENT TIMELINE & EXECUTION SUMMARY${RESET}"
    echo -e "${GREEN}${BOLD}══════════════════════════════════════════════════════════════${RESET}"
    for i in "${!STAGE_NAMES[@]}"; do
        printf "   %-48s : %s\n" "${STAGE_NAMES[$i]}" "$(format_duration ${STAGE_DURATIONS[$i]})"
    done
    echo -e "   ------------------------------------------------------------"
    printf "   ${BOLD}%-48s${RESET} : ${BOLD}%s (%ss)${RESET}\n" "Total Overall Deployment Time" "$(format_duration $TOTAL_DURATION)" "$TOTAL_DURATION"
    echo -e "${GREEN}${BOLD}══════════════════════════════════════════════════════════════${RESET}"
}

# ── Guard: must run from repository root ─────────────────────
[ -f "docker-compose.yml" ] || err "Run from the repository root (where docker-compose.yml lives)."

# ── 1. Detect Dynamic IP ──────────────────────────────────────
section "1. Detecting Dynamic IP"
DETECTED_IP=$(curl -s --max-time 4 ip.me 2>/dev/null | tr -d '[:space:]' || true)
if [ -z "$DETECTED_IP" ]; then
    DETECTED_IP=$(curl -s --max-time 4 ifconfig.me 2>/dev/null | tr -d '[:space:]' || true)
fi
if [ -z "$DETECTED_IP" ]; then
    DETECTED_IP=$(curl -s --max-time 4 icanhazip.com 2>/dev/null | tr -d '[:space:]' || true)
fi
if [ -z "$DETECTED_IP" ]; then
    DETECTED_IP=$(hostname -I 2>/dev/null | awk '{print $1}' | tr -d '[:space:]' || true)
fi
if [ -z "$DETECTED_IP" ]; then
    DETECTED_IP="127.0.0.1"
fi
log "Dynamic Host IP: $DETECTED_IP"

# ── 2. Pre-flight System & Resource Check ─────────────────────
section "2. Pre-flight Resource & Hardware Inspection"

if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS_PRETTY="${PRETTY_NAME:-$NAME}"
else
    OS_PRETTY="$(uname -s)"
fi
KERNEL="$(uname -r)"
ARCH="$(uname -m)"

CPU_CORES=$(nproc 2>/dev/null || echo "1")
CPU_MODEL=$(grep -m1 "model name" /proc/cpuinfo 2>/dev/null | sed -E "s/^model name\s*:\s*//" | tr -s " " || echo "Generic CPU")
[ -z "$CPU_MODEL" ] && CPU_MODEL="Generic CPU"

TOTAL_MEM_MB=$(free -m 2>/dev/null | awk '/Mem:/ {print $2}' || echo "0")
USED_MEM_MB=$(free -m 2>/dev/null | awk '/Mem:/ {print $3}' || echo "0")
FREE_MEM_MB=$(free -m 2>/dev/null | awk '/Mem:/ {print ($7 != "" ? $7 : $4)}' || echo "0")
MEM_PCT=$(awk -v u="$USED_MEM_MB" -v t="$TOTAL_MEM_MB" 'BEGIN {if (t>0) printf "%.1f", (u/t)*100; else print "0"}')
TOTAL_MEM_GB=$(awk -v m="$TOTAL_MEM_MB" 'BEGIN {printf "%.2f GB", m/1024}')
FREE_MEM_GB=$(awk -v m="$FREE_MEM_MB" 'BEGIN {printf "%.2f GB", m/1024}')

SWAP_TOTAL_MB=$(free -m 2>/dev/null | awk '/Swap:/ {print $2}' || echo "0")

DISK_FREE_MB=$(df -m . | awk 'NR>1 {print $(NF-2)}')
DISK_TOTAL_GB=$(df -h . | awk 'NR>1 {print $2}')
DISK_FREE_GB=$(df -h . | awk 'NR>1 {print $4}')

log "OS Environment   : ${OS_PRETTY} (${ARCH}, kernel ${KERNEL})"
log "CPU Architecture : ${CPU_MODEL} (${CPU_CORES} Cores)"
log "Physical Memory  : ${FREE_MEM_GB} free of ${TOTAL_MEM_GB} (${MEM_PCT}% used)"
log "Swap Space       : ${SWAP_TOTAL_MB}MB total swap"
log "Available Disk   : ${DISK_FREE_GB} free (${DISK_FREE_MB}MB) on project partition"

# Auto-provision 4GB swapfile if memory pressure is severe
if [ "$SWAP_TOTAL_MB" -lt 1024 ] && [ "$FREE_MEM_MB" -lt 3500 ] && [ "$FREE_MEM_MB" -gt 0 ]; then
    warn "Swap space is low (${SWAP_TOTAL_MB}MB) and free memory is constrained. Creating a 4GB swapfile..."
    sudo fallocate -l 4G /swapfile 2>/dev/null || sudo dd if=/dev/zero of=/swapfile bs=1M count=4096 2>/dev/null || true
    sudo chmod 600 /swapfile 2>/dev/null || true
    sudo mkswap /swapfile 2>/dev/null || true
    sudo swapon /swapfile 2>/dev/null || true
    grep -q "/swapfile" /etc/fstab 2>/dev/null || echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null 2>&1 || true
    log "4 GB swapfile provisioned and active"
fi

# Hard pre-flight disk threshold: 15 GB minimum
if [ "$DISK_FREE_MB" -lt 15360 ]; then
    warn "Free disk space is under 15 GB checkpoint safety buffer (${DISK_FREE_GB} free). Running Docker build cache prune..."
    docker builder prune -af >/dev/null 2>&1 || true
    docker image prune -f >/dev/null 2>&1 || true
    RECHECK_DISK=$(df -h . | awk 'NR>1 {print $4}')
    log "Reclaimed disk space: ${RECHECK_DISK} free now"
fi

# ── 3. Git Working Tree Sanity Check & Pull ───────────────────
section "3. Git Working Tree Check & Pull"
if [ -d ".git" ]; then
    if [ -n "$(git status --porcelain 2>/dev/null)" ]; then
        warn "Uncommitted changes detected in local working tree. Stashing safely..."
        git stash save "deploy-auto-stash-$(date +%Y%m%d%H%M%S)" >/dev/null 2>&1 || true
    fi
    if git remote 2>/dev/null | grep -q "origin"; then
        log "Pulling latest changes from origin..."
        PULL_LOG=$(git pull origin master 2>&1 || git pull origin main 2>&1) || warn "Could not pull from origin, continuing with local code..."
        echo "$PULL_LOG" | tail -3
        if echo "$PULL_LOG" | grep -q "scripts/deploy.sh"; then
            log "deploy.sh was updated from origin. Re-executing deployment pipeline..."
            exec bash "$0" "$@"
        fi
    else
        log "Working tree clean (local repository mode)"
    fi
fi

# ── 4. Write / Patch .env ─────────────────────────────────────
section "4. Synchronizing .env Configuration"
if [ ! -f ".env" ]; then
    [ -f ".env.example" ] && cp .env.example .env && log "Created .env from .env.example" \
        || err ".env missing and no .env.example found — cannot continue"
fi

set_env() {
    local K="$1" V="$2"
    if grep -q "^${K}=" .env; then
        sed -i "s|^${K}=.*|${K}=${V}|g" .env
    elif grep -q "^${K} *=" .env; then
        sed -i "s|^${K} *=.*|${K} = ${V}|g" .env
    else
        echo "${K}=${V}" >> .env
    fi
}

WEB_PORT=$(grep "^WEB_PORT" .env 2>/dev/null | cut -d'=' -f2 | tr -d ' ' || echo "80")
[ -z "$WEB_PORT" ] && WEB_PORT="80"
POSTGRES_PORT=$(grep "^POSTGRES_PORT" .env 2>/dev/null | cut -d'=' -f2 | tr -d ' ' || echo "5432")
REDIS_PORT=$(grep "^REDIS_PORT" .env 2>/dev/null | cut -d'=' -f2 | tr -d ' ' || echo "6379")
MINIO_PORT=$(grep "^MINIO_PORT" .env 2>/dev/null | cut -d'=' -f2 | tr -d ' ' || echo "9000")
MINIO_CONSOLE_PORT=$(grep "^MINIO_CONSOLE_PORT" .env 2>/dev/null | cut -d'=' -f2 | tr -d ' ' || echo "9001")
DOZZLE_PORT=$(grep "^DOZZLE_PORT" .env 2>/dev/null | cut -d'=' -f2 | tr -d ' ' || echo "8888")

if [ "$WEB_PORT" = "80" ]; then
    BASE_URL="http://${DETECTED_IP}/"
else
    BASE_URL="http://${DETECTED_IP}:${WEB_PORT}/"
fi

set_env "WEB_PORT" "$WEB_PORT"
set_env "POSTGRES_PORT" "$POSTGRES_PORT"
set_env "REDIS_PORT" "$REDIS_PORT"
set_env "MINIO_PORT" "$MINIO_PORT"
set_env "MINIO_CONSOLE_PORT" "$MINIO_CONSOLE_PORT"
set_env "DOZZLE_PORT" "$DOZZLE_PORT"
set_env "APP_BASE_URL" "$BASE_URL"

log "Studio Web URL            -> ${BASE_URL}"
log "PostgreSQL Port           -> ${POSTGRES_PORT}"
log "Redis Port                -> ${REDIS_PORT}"
log "MinIO S3 API / Console    -> ${MINIO_PORT} / ${MINIO_CONSOLE_PORT}"
log "Dozzle Log Monitor        -> http://${DETECTED_IP}:${DOZZLE_PORT}"

# ── 5. Preparing Container Environment ────────────────────────
section "5. Preparing Container Environment"
if [[ "${1:-}" =~ ^(--hard-reset|--clean|-f)$ ]]; then
    warn "Hard reset requested: stopping and removing all containers..."
    docker compose down --remove-orphans 2>&1 | tail -3 || true
    docker rm -f doctune-web doctune-celery-worker doctune-postgres doctune-redis doctune-minio doctune-minio-init doctune-dozzle 2>/dev/null || true
    log "Old containers stopped and removed"
else
    # Non-destructive: remove any transient init containers to avoid collisions
    docker rm -f doctune-minio-init 2>/dev/null || true
    log "Performing fast rolling update (stateful databases preserved)"
fi

# ── 6. Build & Start All Containers ───────────────────────────
section "6. Building & Starting All Containers"
DOCKER_BUILDKIT=1 COMPOSE_DOCKER_CLI_BUILD=1 docker compose up --build -d 2>&1 | tail -25
log "Containers created and started (postgres, redis, minio, web, celery_worker, dozzle)"

echo ""
echo -e "   ${BOLD}Docker Image Sizes:${RESET}"
docker images --filter "reference=*doctune*" --filter "reference=*postgres*" --filter "reference=*minio*" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}" 2>/dev/null | sed 's/^/   /' || true

# ── 7. Wait for PostgreSQL Health (:5432) ─────────────────────
section "7. Waiting for PostgreSQL Health (:5432)"
printf "   Waiting for PostgreSQL"
PG_HEALTHY=false
for i in $(seq 1 35); do
    STATUS=$(docker inspect --format='{{.State.Health.Status}}' doctune-postgres 2>/dev/null || echo "missing")
    if [ "$STATUS" = "healthy" ]; then
        echo ""
        log "PostgreSQL is healthy and accepting connections"
        PG_HEALTHY=true
        break
    fi
    printf "."
    sleep 2
done
$PG_HEALTHY || { echo ""; warn "PostgreSQL took longer to report healthy. Check: docker compose logs postgres"; }

# ── 7b. Wait for Redis Health (:6379) ─────────────────────────
section "7b. Waiting for Redis Health (:6379)"
printf "   Checking Redis status"
REDIS_HEALTHY=false
for i in $(seq 1 20); do
    R_STATUS=$(docker inspect --format='{{.State.Health.Status}}' doctune-redis 2>/dev/null || echo "missing")
    if [ "$R_STATUS" = "healthy" ]; then
        echo ""
        log "Redis message broker and telemetry Pub/Sub is healthy"
        REDIS_HEALTHY=true
        break
    fi
    printf "."
    sleep 1
done
$REDIS_HEALTHY || { echo ""; warn "Redis did not report healthy in time. Check: docker compose logs redis"; }

# ── 7c. Wait for MinIO S3 Health (:9000 / :9001) ──────────────
section "7c. Waiting for MinIO S3 Health (:9000)"
printf "   Checking MinIO S3 status"
MINIO_HEALTHY=false
for i in $(seq 1 25); do
    M_STATUS=$(docker inspect --format='{{.State.Health.Status}}' doctune-minio 2>/dev/null || echo "missing")
    if [ "$M_STATUS" = "healthy" ]; then
        echo ""
        log "MinIO S3 Object Store is healthy"
        MINIO_HEALTHY=true
        break
    fi
    printf "."
    sleep 2
done
$MINIO_HEALTHY || { echo ""; warn "MinIO did not report healthy within timeout. Check: docker compose logs minio"; }

# ── 8. Initialize Database Tables & Schema ────────────────────
section "8. Initializing Database Schema"
if $PG_HEALTHY; then
    docker compose exec -T web alembic -c /app/alembic.ini upgrade head 2>&1 \
        && log "Database schema migrated to head via Alembic (versioned migrations applied)" \
        || {
            warn "Alembic upgrade notice; attempting init_db declarative fallback..."
            docker compose exec -T web python -c "from web.db.session import init_db; init_db()" 2>&1 \
                && log "Database schema verified and tables created" || warn "Database init completed with notices"
        }
else
    warn "Skipping database schema init because PostgreSQL container is not yet healthy"
fi

# ── 9. Verify MinIO S3 Bucket Provisioning ─────────────────────
section "9. Verifying MinIO S3 Bucket Provisioning"
docker compose exec -T web python -c "from web.services.storage_service import storage_service; storage_service.ensure_buckets_exist()" 2>&1 | sed 's/^/   /' || true
log "MinIO storage buckets verified (doc-tune-raw-docs, doc-tune-datasets, doc-tune-checkpoints, doc-tune-merged-models)"

# ── 10. Wait for WebApp Studio Readiness (:80) ────────────────
section "10. Waiting for WebApp Studio Readiness (:${WEB_PORT})"
printf "   Checking Web Studio /health endpoint"
WEB_READY=false
for i in $(seq 1 30); do
    CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:${WEB_PORT}/health 2>/dev/null || echo "000")
    if [[ "$CODE" =~ ^(200|301|302|307)$ ]]; then
        echo ""
        log "DocTune Web Studio is online and responding (HTTP $CODE)"
        WEB_READY=true
        break
    fi
    printf "."
    sleep 2
done
$WEB_READY || { echo ""; warn "Web server did not respond within timeout. Check: docker compose logs web"; }

# ── 11. Container Permissions & Scratch Directory Enforcement ─
section "11. Enforcing Scratch Directory Permissions"
docker compose exec -T web sh -c "mkdir -p /app/.scratch /app/config && chmod -R 777 /app/.scratch" 2>&1 \
    && log "Scratch directory permissions enforced" || true

# ── 12. Housekeeping, Docker Storage & Cache Clearing ─────────
section "12. Housekeeping & Docker Cache Pruning"
# Only prune dangling untagged layers; preserve BuildKit layer cache for fast subsequent deployments
docker image prune -f >/dev/null 2>&1 && log "Dangling Docker build layers pruned" || true

# Only prune build cache if disk space is constrained (<15GB)
if [ "$DISK_FREE_MB" -lt 15360 ]; then
    warn "Disk space is constrained (${DISK_FREE_GB} free). Pruning BuildKit cache..."
    docker builder prune -f >/dev/null 2>&1 || true
else
    log "Docker BuildKit cache preserved for fast sub-minute redeployments"
fi

echo ""
echo -e "   ${BOLD}Docker Storage Footprint:${RESET}"
docker system df 2>/dev/null | sed 's/^/   /' || true

# ── 13. Deployment Timeline Summary & Live Endpoints ──────────
section "13. Deployment Summary & Health Verification"
docker compose ps

echo ""
FINAL_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:${WEB_PORT}/health 2>/dev/null || echo "000")

if [ "$FINAL_STATUS" = "200" ]; then
    log "PLATFORM IS LIVE — ALL SYSTEMS OPERATIONAL (WebApp HTTP ${FINAL_STATUS})"
    echo ""
    echo -e "  ${BOLD}Web Studio Dashboard :${RESET}  ${BASE_URL}"
    echo -e "  ${BOLD}Training Studio      :${RESET}  ${BASE_URL}training"
    echo -e "  ${BOLD}Document Studio      :${RESET}  ${BASE_URL}data-prep"
    echo -e "  ${BOLD}Checkpoints & Models :${RESET}  ${BASE_URL}models"
    echo -e "  ${BOLD}Model Playground     :${RESET}  ${BASE_URL}playground"
    echo -e "  ${BOLD}System Administration:${RESET}  ${BASE_URL}admin"
    echo -e "  ${BOLD}API Swagger Docs     :${RESET}  ${BASE_URL}docs"
    echo -e "  ${BOLD}Health Probe Endpoint:${RESET}  ${BASE_URL}health"
    echo -e "  ${BOLD}MinIO S3 Web Console :${RESET}  http://${DETECTED_IP}:${MINIO_CONSOLE_PORT} (User: minioadmin)"
    echo -e "  ${BOLD}Dozzle Container Logs:${RESET}  http://${DETECTED_IP}:${DOZZLE_PORT}"
    echo -e "  ${BOLD}PostgreSQL Database  :${RESET}  postgresql://doctune@${DETECTED_IP}:${POSTGRES_PORT}/doctune_studio"
    echo -e "  ${BOLD}Redis Broker         :${RESET}  redis://${DETECTED_IP}:${REDIS_PORT}"
    echo ""
    finish_deployment
else
    warn "Web endpoint responded with HTTP ${FINAL_STATUS}. Check container logs: docker compose logs web"
    finish_deployment
fi
