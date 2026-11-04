# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-11-04

### Added
- **Multi-Container Enterprise Orchestration (`docker-compose.yml`)**:
  - `web`: FastAPI Studio & AdminLTE v4 (`Dockerfile.web`, port 4444).
  - `celery_worker`: Distributed ML training & data prep worker (`Dockerfile.worker`, 8GB shm_size, GPU passthrough with CPU fallback).
  - `postgres`: PostgreSQL 16 relational database for run histories, loss curves, and model registry (port 5432).
  - `redis`: Redis 7 broker & real-time telemetry Pub/Sub channel (port 6379).
  - `minio`: MinIO S3 object storage for raw documents, datasets, checkpoints, and merged models (API :9000, Web Console :9001).
  - `minio_init`: Ephemeral bucket auto-provisioner.
- **One-Click Automated Deployment Engine (`scripts/deploy.sh`)**:
  - 13-stage automated pipeline: Dynamic host IP detection, pre-flight system inspection, automated 4GB swapfile creation, git sanity check, `.env` auto-patching, container rebuilds with image/layer size reporting, health probe loops, database schema init, MinIO bucket verification, WebApp readiness check, permission enforcement, build cache pruning, and timeline summary table.
- **Dual-Audience Project Documentation**:
  - User-facing `README.md` ($\le 150$ lines, run-only).
  - Complete software engineering `docs/` hub: architecture overview (C4 context & container diagrams), communication sequences, data & storage ERD, local and Google Cloud Platform (GCP) deployment guide, threat model (STRIDE), API contracts, OpenAPI 3.0 specification (`openapi.yaml`), service guides, runbooks (`restart.md`, `backup-and-restore.md`), and Architecture Decision Records (ADRs).
  - Interactive documentation website configuration (`mkdocs.yml`) and CI verification workflow (`.github/workflows/docs-verify.yml`).
- **Database & Storage Adapters (`web/db/`, `web/services/storage_service.py`)**:
  - SQLAlchemy models for `Run`, `MetricPoint`, `Checkpoint`, `UploadedDocument` with transparent SQLite fallback for native local development.
  - S3 / MinIO storage client with local filesystem fallback.

## [1.0.0] - 2026-11-04

### Added
- **Canonical Pipeline Package (`pipeline/`)**:
  - `pipeline.common`: Centralized `Config` dataclass with `to_dict()`, `from_dict()`, and `load()` support.
  - `pipeline.prepare_data`: Document chunking and teacher LLM dataset generation.
  - `pipeline.train_qlora`: 4-bit QLoRA fine-tuning with structured stdout metric streaming.
  - `pipeline.train_lora`: 16-bit LoRA fine-tuning.
  - `pipeline.train_fft`: Full fine-tuning engine.
  - `pipeline.evaluate`: Fact retrieval, ROUGE, BLEU, and perplexity evaluation.
  - `pipeline.merge_adapters`: Standalone 16-bit model weight merger.
- **Web Studio Engine (`web/`)**:
  - FastAPI asynchronous server running on dedicated default port `4444`.
  - Asynchronous background `ProcessManager` streaming stdout logs over WebSockets.
  - Real-time training metrics parsing (`[METRIC_JSON]`) for interactive loss plotting.
  - Telemetry service inspecting CPU (`/proc/cpuinfo`), RAM (`/proc/meminfo`), Disk (`shutil.disk_usage`), and GPU (`pynvml`).
- **AdminLTE v4 Dashboard & Workflow UI**:
  - Overview Dashboard (`/`) with live job status hero banner, 4-card hardware grid, pipeline stepper, checkpoints registry, and execution history.
  - Document & Data Studio (`/data-prep`) for document upload, chunking, and JSONL dataset preview.
  - Training Studio (`/training`) with real-time ApexCharts loss tracking and log viewer terminal.
  - Checkpoints & Merging (`/models`).
  - Evaluation & Playground (`/playground`) with prompt testing and streaming responses.
- **System Administration Studio (`/admin`)**:
  - Persistent JSON configuration (`config/settings.json`) with atomic `.tmp` file writing and factory reset.
  - Dynamic Application Branding (`server.app_name`, default: `DocTune Studio`) globally rendered across navbar, sidebar, footer, and page title via Jinja2 helpers.
  - Tabbed interface for Server, Hugging Face, Training, Data Prep, and Integrations.
  - Hugging Face token verification API (`/api/admin/test-hf`).
  - Directory verification and creation API (`/api/admin/check-paths`).
  - Webhook alert ping testing (`/api/admin/test-webhook`).
- **Hardware Bounds & Parameter Safeguards**:
  - `get_hardware_bounds()` live host inspection (8 CPU cores, 15 GB RAM, detected GPUs, disk space).
  - Multi-tier boundary clamping preventing CPU/RAM/VRAM overcommitment and Linux OOM kills.
  - Hyperparameter safeguards: batch size [1, 32], gradient accumulation [1, 64], LoRA rank [2, 256], alpha [2, 512], epochs [1, 50], sequence length [128, 8192], chunk overlap <= 50% chunk size.
  - Pre-flight storage safety check enforcing minimum 15 GB free disk before launching training jobs.
  - OS-level subprocess environment injection (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `CUDA_VISIBLE_DEVICES`).

### Changed
- Reorganized project root: removed 7 forwarder scripts in favor of `pipeline/` package execution.
- Standardized navigation bar and sidebar on AdminLTE v4 with dark/light mode toggle.
- Synchronized all studio views (`dashboard`, `training`, `models`, `playground`) to load persistent settings from `config/settings.json` via `Config.load()`.
