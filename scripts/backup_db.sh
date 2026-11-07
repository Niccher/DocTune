#!/usr/bin/env bash
# ==============================================================================
# scripts/backup_db.sh — Automated PostgreSQL Database Backup Utility
#                         for DocTune Studio
#
# Usage:
#   bash scripts/backup_db.sh [--keep-days 7] [--s3-upload]
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

BACKUP_DIR="${BACKUP_DIR:-$PROJECT_ROOT/backups}"
KEEP_DAYS=7
S3_UPLOAD=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --keep-days)
            KEEP_DAYS="$2"
            shift 2
            ;;
        --s3-upload)
            S3_UPLOAD=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
mkdir -p "$BACKUP_DIR"
BACKUP_FILE="$BACKUP_DIR/doctune_studio_${TIMESTAMP}.sql.gz"

echo "=== DocTune Database Backup ==="
echo "Timestamp    : $TIMESTAMP"
echo "Destination  : $BACKUP_FILE"

# 1. Load database credentials from .env if present
POSTGRES_USER="doctune"
POSTGRES_DB="doctune_studio"
if [ -f .env ]; then
    VAL_USER=$(grep -E '^POSTGRES_USER=' .env | cut -d '=' -f2- | tr -d '"' | tr -d "'" || true)
    VAL_DB=$(grep -E '^POSTGRES_DB=' .env | cut -d '=' -f2- | tr -d '"' | tr -d "'" || true)
    [ -n "$VAL_USER" ] && POSTGRES_USER="$VAL_USER"
    [ -n "$VAL_DB" ] && POSTGRES_DB="$VAL_DB"
fi

# 2. Execute pg_dump from container
if docker compose ps postgres --filter "status=running" -q >/dev/null 2>&1; then
    docker compose exec -T postgres pg_dump -U "$POSTGRES_USER" "$POSTGRES_DB" | gzip > "$BACKUP_FILE"
    SIZE=$(du -h "$BACKUP_FILE" | cut -f1)
    echo "[OK] Backup completed successfully ($SIZE): $BACKUP_FILE"
else
    echo "[FAIL] PostgreSQL container is not running. Aborting backup." >&2
    exit 1
fi

# 3. Optional S3 / MinIO upload
if $S3_UPLOAD; then
    echo "Uploading backup to MinIO/S3..."
    docker compose exec -T web python -c "
from pathlib import Path
from web.services.storage_service import storage_service
backup_path = Path('$BACKUP_FILE')
if backup_path.exists():
    storage_service.upload_file(backup_path, storage_service.bucket_checkpoints, 'backups/' + backup_path.name)
    print('[OK] Uploaded to S3: backups/' + backup_path.name)
" 2>&1 || echo "[WARN] S3 backup upload skipped"
fi

# 4. Prune old backups
if [ "$KEEP_DAYS" -gt 0 ]; then
    echo "Pruning backups older than $KEEP_DAYS days in $BACKUP_DIR..."
    find "$BACKUP_DIR" -type f -name "doctune_studio_*.sql.gz" -mtime +"$KEEP_DAYS" -exec rm -f {} +
fi

echo "=== Backup Finished ==="
