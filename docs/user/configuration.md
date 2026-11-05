# Configuration Reference

All user-configurable options are managed via the `.env` file at the repository root.

---

## Environment Variables Dictionary

| Variable | Service | Required | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `APP_NAME` | `web` | No | `"DocTune Studio"` | Global application title rendered across UI header, navbar, footer, and page title |
| `WEB_PORT` | `web` | Yes | `80` | External HTTP host port for FastAPI and AdminLTE studio (standard HTTP) |
| `POSTGRES_DB` | `postgres` | Yes | `doctune_studio` | Database name |
| `POSTGRES_USER` | `postgres` | Yes | `doctune` | Database master user |
| `POSTGRES_PASSWORD` | `postgres` | Yes | `doctune_secure_password_2026` | Database password |
| `POSTGRES_PORT` | `postgres` | Yes | `5432` | Host port for PostgreSQL |
| `REDIS_PORT` | `redis` | Yes | `6379` | Host port for Redis message broker |
| `MINIO_ROOT_USER` | `minio` | Yes | `minioadmin` | S3 administrative user |
| `MINIO_ROOT_PASSWORD`| `minio` | Yes | `minioadmin_secure_password_2026`| S3 administrative password |
| `MINIO_PORT` | `minio` | Yes | `9000` | S3 API endpoint port |
| `MINIO_CONSOLE_PORT`| `minio` | Yes | `9001` | MinIO web console port |
| `DOZZLE_PORT` | `dozzle` | Yes | `8888` | Host port for Dozzle real-time container log viewer |
| `HF_TOKEN` | `celery_worker`| Cond. | *(empty)* | Hugging Face user access token (required for Meta-Llama models) |
| `MAX_CPU_THREADS` | `celery_worker`| No | `7` | Ceiling for OpenMP / Torch thread pool |
| `MAX_RAM_GB` | `celery_worker`| No | `13` | Maximum host RAM allocation in GB |
| `S3_BUCKET_DOCS` | `web` | Yes | `doc-tune-raw-docs` | S3 bucket for incoming PDFs/DOCX |
| `S3_BUCKET_DATASETS` | `web` | Yes | `doc-tune-datasets` | S3 bucket for processed JSONL datasets |
| `S3_BUCKET_CHECKPOINTS`| `web`| Yes | `doc-tune-checkpoints` | S3 bucket for LoRA/QLoRA adapter weights |
| `S3_BUCKET_MERGED` | `web` | Yes | `doc-tune-merged-models` | S3 bucket for 16-bit standalone merged models |
