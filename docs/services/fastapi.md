# FastAPI Web Studio Service

The `web` service is a high-performance asynchronous web server built with **FastAPI** and styled with **AdminLTE v4** (Bootstrap 5).

---

## 1. Directory Structure

```
web/
├── db/                         ← SQLAlchemy models & database session
│   ├── models.py
│   └── session.py
├── routers/                    ← Modular FastAPI route handlers
│   ├── admin.py                ← /admin & configuration APIs
│   ├── dashboard.py            ← / overview dashboard
│   ├── data_prep.py            ← /data-prep document ingestion
│   ├── models.py               ← /models checkpoint registry
│   ├── playground.py           ← /playground prompt testing
│   └── training.py             ← /training real-time loss graph & controls
├── services/                   ← Core business services
│   ├── config_service.py       ← settings.json persistence & clamping
│   ├── process_manager.py      ← Async subprocess lifecycle & reader stream
│   ├── storage_service.py      ← MinIO / S3 object store client
│   └── system_service.py       ← Hardware telemetry & boundary inspection
├── static/                     ← Vendor assets, CSS, JS
├── templates/                  ← Jinja2 templates (base.html, admin.html, etc.)
├── main.py                     ← FastAPI application entrypoint
└── templates_config.py         ← Centralized Jinja2 global helpers (get_app_name)
```

---

## 2. Running Locally (Native Development)

```bash
# Activate virtual environment
source .venv/bin/activate

# Run with auto-reload enabled on port 4444
uvicorn web.main:app --reload --host 0.0.0.0 --port 4444
```
