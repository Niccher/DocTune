# FastAPI Web Studio Service

The `web` service is a high-performance asynchronous web server built with **FastAPI** and styled with **AdminLTE v4** (Bootstrap 5).

---

## 1. Directory Structure

```
web/
├── db/                         ← SQLAlchemy models & database session
│   ├── models.py               ← Run, MetricPoint, Checkpoint, DatasetVersion, etc.
│   └── session.py              ← Engine, SessionLocal, auto-migrations
├── routers/                    ← Modular FastAPI route handlers
│   ├── admin.py                ← /admin & hardware clamping APIs
│   ├── dashboard.py            ← / overview dashboard
│   ├── data_prep.py            ← /data-prep document ingestion & dataset snapshots
│   ├── experiments.py          ← /experiments run comparison & RAG benchmark APIs
│   ├── inference.py            ← /v1/chat/completions OpenAI-compatible inference
│   ├── models.py               ← /models checkpoint registry, GGUF export, HF Hub push
│   ├── playground.py           ← /playground prompt testing & DPO alignment
│   └── training.py             ← /training real-time loss graph, queue & sweeps
├── services/                   ← Core business services
│   ├── config_service.py       ← settings.json persistence & boundary clamping
│   ├── dataset_version_service.py ← Immutable dataset snapshotting & SHA-256 fingerprinting
│   ├── hf_model_service.py     ← HF Hub downloads, caching & private repo push
│   ├── inference_service.py    ← OpenAI-compatible streaming chat completion engine
│   ├── process_manager.py      ← Async subprocess lifecycle & reader stream
│   ├── queue_manager.py        ← FIFO training job queue & resumable checkpoints
│   ├── storage_service.py      ← MinIO / S3 object store client with local fallback
│   ├── sweep_service.py        ← Hyperparameter grid sweep combinatorial generator
│   ├── system_service.py       ← Hardware telemetry & boundary inspection
│   └── teacher_service.py      ← Universal Teacher LLM adapter & connection testing
├── static/                     ← Vendor assets, CSS, JS
├── templates/                  ← Jinja2 templates (base.html, experiments.html, etc.)
├── main.py                     ← FastAPI application entrypoint
└── templates_config.py         ← Centralized Jinja2 global helpers (get_app_name)
```

---

## 2. Running Locally (Native Development)

```bash
# Activate virtual environment
source .venv/bin/activate

# Run with auto-reload enabled (default port 80, or 8080 for unprivileged local dev)
python run_web.py --reload
```
