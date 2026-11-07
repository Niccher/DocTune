# Testing Guide & Quality Assurance

DocTune Studio maintains an automated regression test suite in [`tests/test_web.py`](../../tests/test_web.py).

---

## 1. Running Automated Tests

Run the test suite directly with Python:

```bash
# Execute all automated tests
.venv/bin/python tests/test_web.py
```

Expected output:
```text
Running tests directly...
All tests passed successfully!
```

---

## 2. Test Coverage

The 63-test automated regression suite validates:
* **Route Rendering**: All 7 studio views (Dashboard, Data Prep, Training, Run Comparison `/experiments`, Checkpoints/Models, Playground, Admin).
* **System Telemetry & Hardware Bounds**: Host resource sampling, CPU/RAM/GPU boundary clamping, and port 80/8080 checks.
* **Storage & S3 Integration**: MinIO S3 object storage upload, directory syncing, presigned URL generation, and local fallback.
* **Database & Migrations**: Auto-column schema upgrades, Alembic baseline checks, and rehydration from DB on server reboot.
* **Data Preparation & Ingestion**: Teacher LLM discovery, connection diagnostics, synthetic Q&A extraction, and PII redaction preview (`POST /api/v1/data-prep/preview-redaction`).
* **Dataset Versioning & Provenance**: Immutable snapshotting, SHA-256 fingerprinting, manifest tracking, and active dataset switching (`GET /api/v1/dataset/versions`, `POST /api/v1/dataset/versions/activate`).
* **Training Queue & Checkpoint Resumption**: FIFO queue ordering, auto-advancing on job completion, cancellation, and `--resume-from-checkpoint` state restoration across LoRA, QLoRA, and FFT.
* **Hyperparameter Sweeps**: Combinatorial grid generation, Cartesian parameter expansion, trial enqueuing, and limit clamping (`POST /api/v1/training/sweep/start`).
* **Run Comparison Studio**: Multi-series step-loss telemetry curve alignment and hyperparameter diff tables (`GET /api/v1/experiments/runs`, `GET /api/v1/experiments/compare`).
* **Hugging Face Hub Integration**: Base model search, cache deduplication, background weight downloads, and private repository publishing (`POST /api/v1/models/push-to-hub`).
* **Empirical RAG Benchmark**: 4-way evaluation harness measuring Factual Accuracy, Hallucination/Refusal Rate, Latency, and Token Overhead (`POST /api/v1/experiments/rag-benchmark`).
* **GGUF Export & Ollama Registration**: Quantization, Modelfile generation, and local Ollama daemon registration (`POST /api/v1/models/export-gguf`, `POST /api/v1/models/ollama/register`).
* **OpenAI-Compatible Inference API**: Standard `/v1/chat/completions` non-streaming and streaming Server-Sent Events (SSE), and `/v1/models` catalog.
* **Human Feedback Alignment (DPO)**: Thumbs up/down rating capture, preference pair export (`POST /api/v1/playground/dpo/export`), and TRL DPO training dispatch.
