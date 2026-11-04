# Runbook: Backup & Disaster Recovery

Procedures for snapshotting PostgreSQL databases and archiving MinIO S3 object buckets.

---

## 1. Automated PostgreSQL Snapshot

```bash
# Create timestamped SQL dump
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
docker compose exec -T postgres pg_dump -U doctune doctune_studio > "backup_pg_${TIMESTAMP}.sql"
echo "PostgreSQL backup created: backup_pg_${TIMESTAMP}.sql"
```

### Restoring PostgreSQL from Snapshot
```bash
# Restore SQL dump into container
cat "backup_pg_${TIMESTAMP}.sql" | docker compose exec -T postgres psql -U doctune -d doctune_studio
```

---

## 2. MinIO S3 Checkpoint Archival

Use the `mc` (MinIO Client) command line inside the container or host to mirror buckets:

```bash
# Mirror all checkpoints to a local archive directory
docker compose exec -T minio_init mc mirror myminio/doc-tune-checkpoints /tmp/checkpoints_archive
```
