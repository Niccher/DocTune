# Architecture Overview

DocTune Studio is structured as a **distributed, cloud-native ML platform**. Heavy computations (gradient computation, quantization, document tokenization) are strictly decoupled from the web application layer to ensure high responsiveness and fault tolerance.

---

## 1. C4 Context Diagram

```mermaid
flowchart TB
    User["Operator / ML Engineer"]
    HF["Hugging Face Hub\n(Base Models & Tokenizers)"]
    Teacher["Universal Teacher LLM Gateway\n(Ollama / llama.cpp / Grok / Gemini / OpenAI)"]

    subgraph Studio["DocTune Studio Platform"]
        WebUI["Web Studio & AdminLTE v4\n(FastAPI :80)"]
        Worker["Distributed Celery Worker\n(PyTorch / CUDA / PEFT)"]
        Storage[("MinIO S3 Object Storage\n(:9000 / :9001)")]
        DB[("PostgreSQL 16 Database\n(:5432)")]
        Broker[("Redis 7 Broker & Pub/Sub\n(:6379)")]
    end

    User -->|Web Browser / REST API| WebUI
    WebUI -->|Dispatch Training Tasks| Broker
    Broker -->|Consume Tasks| Worker
    Worker -->|Download Base Weights| HF
    Worker -->|Synthetic Q&A Ingestion| Teacher
    Worker -->|Stream [METRIC_JSON]| Broker
    Broker -.->|Pub/Sub Telemetry| WebUI
    Worker -->|Save Checkpoints & Weights| Storage
    WebUI -->|Record Runs & Metadata| DB
```

---

## 2. Container Mapping

| Container Name | Base Image | Service Responsibility | Exposed Ports | Persistent Volume |
| :--- | :--- | :--- | :--- | :--- |
| `doctune-web` | `python:3.12-slim` | Serves AdminLTE UI, REST API, WebSocket loss streaming | `80` | `./config:/app/config`, `scratch_data` |
| `doctune-celery-worker` | `python:3.11-slim` | Executes training, tokenization, checkpoint uploads | None | `hf_cache`, `scratch_data` |
| `doctune-postgres` | `postgres:16-alpine`| Relational store for runs, metrics, checkpoint metadata | `5432` | `pg_data` |
| `doctune-redis` | `redis:7-alpine` | Celery task queue & real-time metric Pub/Sub channel | `6379` | `redis_data` |
| `doctune-minio` | `cgr.dev/chainguard/minio:latest`| S3-compatible object storage for heavy model weights | `9000, 9001` | `minio_data` |
| `doctune-dozzle` | `amir20/dozzle:latest`| Real-time container log monitoring & streaming | `8888` | `/var/run/docker.sock` |

---

## 3. Trust Boundaries & Security Model

1. **Client to Web Studio (`:80`)**: Standard HTTP/WebSocket traffic. All hyperparameter inputs are sanitized and clamped at the boundary before execution.
2. **Internal Network (`doctune-net`)**: Private Docker bridge network. PostgreSQL and Redis endpoints are accessible only to `web` and `celery_worker` containers.
3. **Artifact Isolation**: Model weights are stored in MinIO and served via time-limited **S3 Presigned URLs**, preventing direct filesystem access from external clients.
