# DocTune Studio

DocTune Studio is a private document fine-tuning platform for training 8B-class open-source LLMs (Llama 3, Mistral, Qwen) on proprietary enterprise documents using 4-bit QLoRA, 16-bit LoRA, and Full Fine-Tuning.

Stack: Python FastAPI, AdminLTE v4, PyTorch, Celery, Redis, PostgreSQL, MinIO S3.

**If you only need to run the system, this page is enough.**  
Software engineers and contributors: see [docs/README.md](docs/README.md).

## What “Running” Looks Like

| Component | URL / Endpoint | Purpose |
| :--- | :--- | :--- |
| **Web Studio UI** | `http://localhost` | Interactive pipeline dashboard, loss curves, model playground |
| **MinIO S3 Console** | `http://localhost:9001` | Object storage browser (User: `minioadmin`, Pass: `minioadmin_secure_password_2026`) |
| **FastAPI Swagger** | `http://localhost/docs` | Interactive REST API documentation |
| **Dozzle Log Monitor**| `http://localhost:8888` | Real-time container log monitoring & streaming |
| **Health Probe** | `GET http://localhost/health` | Container readiness verification (HTTP 200) |
| **Redis Broker** | `redis://localhost:6379` | In-memory message broker & Pub/Sub telemetry stream |
| **PostgreSQL DB** | `postgresql://localhost:5432` | Relational storage for runs, metrics, and checkpoints |

## Prerequisites

### Option A — Docker Compose (Recommended)
- Docker Engine 24+ & Docker Compose v2 (or Docker Desktop)
- 15 GB free disk space (for model checkpoints and datasets)
- Optional: NVIDIA GPU with NVIDIA Container Toolkit (defaults to CPU fallback if absent)

### Option B — Without Docker (Native Python)
- Python 3.11 or 3.12, Git, virtualenv

## Setup and Run

### 1. One-Click Deployment (Recommended)
From the repository root, run the automated 13-stage deployment pipeline:

```bash
bash scripts/deploy.sh
```

The script inspects host CPU/RAM/disk, auto-creates swap if constrained, configures `.env`, builds all containers, runs database migrations, provisions MinIO buckets, and verifies system health.

### 2. Manual Docker Compose Run
If you prefer running compose commands directly:

```bash
# 1. Create environment file
cp .env.example .env

# 2. Build and start all services
docker compose up --build -d

# 3. View live logs
docker compose logs -f web

# 4. Stop containers
docker compose down
```

### 3. Native CLI Mode (Without Docker)
```bash
# 1. Setup virtual environment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-web.txt -r requirements.txt

# 2. Launch local web studio (port 80 / fallback 8080)
python run_web.py
```

## Configuration Users May Change

Edit `.env` before running `bash scripts/deploy.sh`:

| Variable | Service | Default | Description |
| :--- | :--- | :--- | :--- |
| `WEB_PORT` | `web` | `80` | Web Studio host port (standard HTTP) |
| `HF_TOKEN` | `celery_worker` | *(empty)* | Hugging Face access token for gated models (e.g. Llama 3) |
| `MAX_CPU_THREADS` | `celery_worker` | `7` | Maximum CPU thread pool (leaving 1 thread for OS) |
| `MAX_RAM_GB` | `celery_worker` | `13` | RAM allocation budget ceiling in GB |
| `MINIO_PORT` | `minio` | `9000` | S3 API endpoint port |
| `MINIO_CONSOLE_PORT`| `minio` | `9001` | MinIO web console port |

Full environment specification: [docs/user/configuration.md](docs/user/configuration.md).

## Something Went Wrong?

- **Port 80 or 5432 in use**: Update `WEB_PORT` or `POSTGRES_PORT` in `.env` and re-run `bash scripts/deploy.sh`.
- **Low disk space error**: DocTune requires $\ge 15\text{ GB}$ free disk for checkpoints. Run `docker system prune -af` to reclaim space.
- **Out of Memory during training**: Lower `per_device_train_batch_size` to 1 or 2, and switch method to `QLoRA (4-bit)`.
- Full troubleshooting guide: [docs/user/troubleshooting.md](docs/user/troubleshooting.md).

## Software Engineers

- Architecture Overview & C4 Diagrams: [docs/architecture/overview.md](docs/architecture/overview.md)
- Inter-Service Communication & Sequences: [docs/architecture/communication.md](docs/architecture/communication.md)
- Google Cloud Platform (GCP) Deployment: [docs/architecture/deployment.md](docs/architecture/deployment.md)
- Making Changes & Contribution Matrix: [docs/engineering/making-changes.md](docs/engineering/making-changes.md)
