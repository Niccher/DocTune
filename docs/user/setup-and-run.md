# Setup and Run Guide

This guide walks operators through provisioning and running DocTune Studio from a fresh machine.

---

## 1. Automated Deployment via `deploy.sh` (Recommended)

DocTune Studio includes a 13-stage automated deployment pipeline in [`scripts/deploy.sh`](../../scripts/deploy.sh).

```bash
# Clone the repository
git clone https://github.com/Niccher/Fine-Tuning.git
cd "Fine-Tuning"

# Execute automated deployment pipeline
bash scripts/deploy.sh
```

### What `deploy.sh` Does Automatically:
1. **Dynamic IP Detection**: Automatically detects your public IP or primary LAN interface so all services bind cleanly.
2. **Resource Inspection**: Queries CPU architecture, RAM, swap space, and disk capacity. If available swap is low (<1 GB) and free RAM is <3.5 GB, it automatically provisions a **4 GB swapfile** to prevent system freeze.
3. **Environment Patching**: Generates `.env` from `.env.example` with valid credentials and standard ports.
4. **Container Build & Launch**: Builds images with BuildKit and launches `postgres`, `redis`, `minio`, `minio_init`, `web`, and `celery_worker`.
5. **Health Verification**: Polls PostgreSQL, Redis, MinIO, and FastAPI `/health` endpoints with animated retry loops.
6. **Bucket & Schema Initialization**: Executes table creation and provisions required MinIO S3 buckets.
7. **Housekeeping & Timeline Summary**: Prunes dangling build layers and displays a formatted stage-by-stage execution summary.

---

## 2. Verifying a Healthy Deployment

Run `docker compose ps` to inspect running services:

```bash
$ docker compose ps
NAME                     IMAGE                              COMMAND                  SERVICE          CREATED         STATUS                   PORTS
doctune-postgres         postgres:16-alpine                 "docker-entrypoint.s…"   postgres         2 minutes ago   Up 2 minutes (healthy)   0.0.0.0:5432->5432/tcp
doctune-redis            redis:7-alpine                     "docker-entrypoint.s…"   redis            2 minutes ago   Up 2 minutes (healthy)   0.0.0.0:6379->6379/tcp
doctune-minio            cgr.dev/chainguard/minio:latest    "/opt/bin/minio serv…"   minio            2 minutes ago   Up 2 minutes (healthy)   0.0.0.0:9000->9000/tcp, 0.0.0.0:9001->9001/tcp
doctune-web              doctune-web                        "python run_web.py -…"   web              2 minutes ago   Up 2 minutes (healthy)   0.0.0.0:80->80/tcp
doctune-celery-worker    doctune-celery-worker              "celery -A web.celer…"   celery_worker    2 minutes ago   Up 2 minutes             
doctune-dozzle           amir20/dozzle:latest               "/dozzle"                dozzle           2 minutes ago   Up 2 minutes             0.0.0.0:8888->8080/tcp
```

---

## 3. Stopping and Restarting

```bash
# Graceful shutdown (preserves all database and S3 checkpoint volumes)
docker compose down

# Start containers again without rebuilding
docker compose up -d

# View real-time logs
docker compose logs -f web celery_worker
```
