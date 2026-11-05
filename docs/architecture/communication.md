# Communication Protocols & Sequences

Details of inter-container communication, protocol schemas, and end-to-end task flows.

---

## 1. Inter-Service Communication Matrix

| Source | Destination | Transport | Protocol | Auth | Purpose |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Browser** | `doctune-web` | TCP | HTTP/1.1 | Session / Local | UI pages, REST API requests |
| **Browser** | `doctune-web` | TCP | WebSocket | None (Local) | Real-time log & loss curve streaming |
| **Browser** | `doctune-minio` | TCP | HTTPS/S3 | Presigned URL | Direct model artifact download |
| `doctune-web` | `doctune-postgres` | TCP | PostgreSQL Wire | User/Password | Reading & persisting runs and metadata |
| `doctune-web` | `doctune-redis` | TCP | RESP | None (Internal) | Enqueuing Celery tasks & Pub/Sub subscription |
| `doctune-worker`| `doctune-redis` | TCP | RESP | None (Internal) | Consuming tasks & publishing metrics |
| `doctune-worker`| `doctune-minio` | TCP | S3 REST (boto3) | S3 v4 HMAC | Uploading checkpoints & downloading datasets |
| `doctune-web` / `worker` | **Teacher LLM Gateway** | TCP | HTTP/HTTPS | Bearer Key / None | Synthetic Q&A dataset generation & connection validation |

---

## 2. Training Sequence & Real-Time Loss Streaming

```mermaid
sequenceDiagram
    autonumber
    actor User as Operator / Browser
    participant Web as FastAPI Studio (:80)
    participant DB as PostgreSQL (:5432)
    participant Redis as Redis Pub/Sub (:6379)
    participant Worker as Celery ML Worker
    participant MinIO as MinIO S3 (:9000)

    User->>Web: POST /api/v1/training/start (method, epochs, batch_size, lora_r)
    Web->>DB: INSERT INTO runs (status='PENDING')
    Web->>Redis: celery.send_task("task_train_model", run_id, config)
    Web-->>User: 200 OK {"status": "started", "job_id": run_id}

    User->>Web: Connect WebSocket /ws/stream
    Web->>Redis: SUBSCRIBE "run:{run_id}:stream"

    Worker->>Redis: BRPOP training_queue
    Worker->>DB: UPDATE runs SET status='RUNNING'
    Worker->>Worker: Execute PyTorch QLoRA Fine-Tuning Step

    loop Every Training Step
        Worker->>Redis: PUBLISH "run:{run_id}:stream" {"step": 10, "loss": 1.42, "epoch": 0.25}
        Redis-->>Web: Message {"step": 10, "loss": 1.42}
        Web-->>User: WS Frame {"step": 10, "loss": 1.42}
        Note over User: ApexCharts plots point dynamically
    end

    Worker->>MinIO: PUT s3://doc-tune-checkpoints/{run_id}/adapter_model.safetensors
    Worker->>DB: UPDATE runs SET status='COMPLETED', completed_at=NOW()
    Worker->>Redis: PUBLISH "run:{run_id}:stream" {"status": "COMPLETED"}
    Redis-->>Web: Message {"status": "COMPLETED"}
    Web-->>User: WS Frame {"status": "COMPLETED"}
```
