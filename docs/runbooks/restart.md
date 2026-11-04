# Runbook: Container Restart & Incident Recovery

Standard operating procedures for restarting containers, recovering from deadlocks, and performing clean rebuilds without losing database volumes.

---

## 1. Clean Rebuild Without Data Loss

To pull the latest code and rebuild application images while keeping all PostgreSQL databases and MinIO checkpoint weights intact:

```bash
# 1. Stop active containers
docker compose down

# 2. Rebuild images with no cache
docker compose build --no-cache web celery_worker

# 3. Start stack in background
docker compose up -d

# 4. Confirm container health
docker compose ps
```

---

## 2. Nuclear Reset (Wipe All State)

> [!CAUTION]
> This command will permanently delete all training history, database records, and stored MinIO checkpoints!

```bash
# Stop containers and destroy all named volumes
docker compose down -v

# Relaunch from scratch
bash scripts/deploy.sh
```
