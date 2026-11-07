# REST & WebSocket API Contract Reference

DocTune Studio exposes a canonical, versioned REST API under `/api/v1/*` and WebSocket streams for external automation, pipeline orchestration, and dashboard integrations.

> **Versioning Policy**: All domain REST endpoints are strictly routed under `/api/v1/*`. Legacy unversioned `/api/*` endpoints return `404 Not Found` (no backward compatibility aliases). System operational probes (`/health`) and real-time streaming sockets (`/ws/*`) remain unversioned at root.

---

## 1. System & Health Probes

### `GET /health`
Container and load balancer healthcheck endpoint.
* **Response**: `200 OK`
  ```json
  {
    "status": "healthy",
    "app": "DocTune Studio",
    "version": "1.0.0",
    "s3_storage": true
  }
  ```

---

## 2. Dashboard & Telemetry

### `GET /api/v1/telemetry`
Returns live host resource metrics (CPU, RAM, GPU, Disk) and pipeline status.
* **Response**: `200 OK`
  ```json
  {
    "system": {
      "cpu": {"percent": 18.2, "count": 8},
      "ram": {"percent": 45.1, "used_gb": 6.8, "total_gb": 15.1},
      "disk": {"percent": 32.0, "free_gb": 133.2},
      "gpu": {"available": false, "name": "N/A"}
    },
    "pipeline": {
      "status": "IDLE",
      "active_job": null,
      "recent_jobs": [],
      "checkpoints": []
    }
  }
  ```

---

## 3. Data Preparation & Teacher LLM Endpoints

### `GET /api/v1/teacher/presets`
Returns supported teacher inference engines (`ollama`, `llamacpp`, `vllm`, `gemini`, `deepseek`, `grok`, `openai`, `anthropic`, `custom`) and their base URL / model configurations.

### `GET /api/v1/teacher/models?provider={provider}&base_url={url}&api_key={key}`
Dynamically queries a teacher endpoint via standard `/v1/models` to discover hosted model IDs.

### `POST /api/v1/teacher/test`
Validates connectivity, measuring latency and reporting available model weights.
* **Payload**:
  ```json
  {
    "provider": "ollama",
    "base_url": "http://localhost:11434/v1",
    "api_key": "not-needed",
    "model": "llama3.1:latest"
  }
  ```
* **Response**:
  ```json
  {
    "valid": true,
    "latency_ms": 38,
    "models": ["llama3.1:latest"],
    "active_model": "llama3.1:latest",
    "message": "Connection verified successfully!"
  }
  ```

### `POST /api/v1/documents/upload`
Uploads raw documents (`.pdf`, `.docx`, `.txt`, `.md`) to raw storage or S3.
* **Content-Type**: `multipart/form-data` (`files`)

### `POST /api/v1/documents/delete`
Deletes an uningested document from raw storage.
* **Form Parameters**: `filename`

### `POST /api/v1/data-prep/start`
Starts background synthetic QA dataset extraction pipeline.
* **Form Parameters**: `chunk_size`, `chunk_overlap`, `provider`, `teacher_base_url`, `teacher_model`, `teacher_api_key`.

### `GET /api/v1/dataset/preview?offset=0&limit=5`
Paginated inspection of generated instruction-tuning dataset pairs (`train.jsonl`).

---

## 4. Training Studio Endpoints

### `POST /api/v1/training/start`
Starts a QLoRA / LoRA / Full fine-tuning run.
* **Form Parameters**:
  * `method` (string): `qlora`, `lora`, or `fft`. Default: `qlora`.
  * `model_id` (string): Base model repository ID.
  * `epochs` (integer): Training epochs (clamped [1, 50]).
  * `batch_size` (integer): Per-device batch size (clamped [1, 32]).
  * `grad_accum` (integer): Gradient accumulation steps (clamped [1, 64]).
  * `lora_r` (integer): LoRA attention rank (clamped [2, 256]).
  * `lora_alpha` (integer): LoRA scaling factor (clamped [2, 512]).
  * `learning_rate` (float): Learning rate (clamped [1e-7, 0.1]).
  * `max_seq_length` (integer): Context token limit (clamped [128, 8192]).
* **Response**: `200 OK`
  ```json
  {
    "status": "started",
    "job_id": "run_1730732800"
  }
  ```

### `POST /api/v1/training/stop`
Aborts currently active training or data preparation process.
* **Response**: `200 OK`
  ```json
  {
    "status": "success",
    "message": "Job abort signal sent"
  }
  ```

---

## 5. Model Management & Evaluation

### `POST /api/v1/models/merge`
Fuses 16-bit LoRA adapter weights with base checkpoint.
* **Form Parameters**: `adapter_path`, `output_name`, `export_format` (`hf` or `gguf`).

### `POST /api/v1/evaluate/run`
Runs automated validation benchmarks against test split.

### `POST /api/v1/playground/chat`
Performs side-by-side inference queries comparing base and fine-tuned checkpoints.
* **Form Parameters**: `prompt`

---

## 6. System Administration Endpoints

### `GET /api/v1/admin/hardware-bounds`
Returns physical host resource boundaries and safe ceilings.
* **Response**: `200 OK`
  ```json
  {
    "host_cores": 8,
    "max_safe_cpu": 7,
    "total_ram_gb": 15.06,
    "max_safe_ram": 13,
    "detected_gpus": 0,
    "max_safe_gpus": 0,
    "free_disk_gb": 133.2,
    "min_required_disk_gb": 15.0,
    "has_sufficient_disk": true
  }
  ```

### `GET /api/v1/admin/settings`
Returns active `config/settings.json`.

### `POST /api/v1/admin/settings`
Updates and persists system configurations with automatic physical boundary clamping.

### `POST /api/v1/admin/reset`
Restores configurations to factory defaults.

### `GET /api/v1/admin/export`
Downloads `fine_tuning_settings.json`.

### `POST /api/v1/admin/test-hf`
Validates Hugging Face access tokens against HF Hub.

### `POST /api/v1/admin/check-paths`
Verifies and creates pipeline directory paths.

### `POST /api/v1/admin/test-webhook`
Sends a test notification ping to alert webhook.

---

## 7. WebSocket Streaming

### `WS /ws/stream` & `WS /ws/training`
Full-duplex WebSocket stream for real-time loss telemetry and terminal logging.

* **Client Message**: Send `"ping"` to receive `"pong"` heartbeat.
* **Server Frames**:
  * Raw terminal lines:
    ```
    Epoch 1/3: 25% [=====>     ] loss=1.423
    ```
  * Structured metrics:
    ```json
    "[METRIC_JSON] {\"step\": 20, \"loss\": 1.35, \"epoch\": 0.4}"
    ```

---

## 8. OpenAI-Compatible Inference API

Standard drop-in inference endpoints allowing external tools, LangChain, LlamaIndex, or OpenAI SDKs to interact with trained DocTune models.

### `POST /v1/chat/completions`
Generates chat completions with optional streaming.
* **Payload**:
  ```json
  {
    "model": "doctune-merged",
    "messages": [
      {"role": "user", "content": "What is the primary failover protocol for Facility B?"}
    ],
    "temperature": 0.7,
    "max_tokens": 512,
    "stream": false
  }
  ```
* **Streaming**: When `stream: true`, responses are emitted as standard Server-Sent Events (`data: {"choices": [{"delta": {"content": "..."}}]}`).

### `GET /v1/models`
Returns list of available models and merged checkpoints in OpenAI catalog format.

---

## 9. Experiments, Run Comparison & RAG Benchmarks

### `GET /api/v1/experiments/runs?limit=50`
Returns recent training runs decorated with hyperparameters, final loss, status, duration, and dataset version tags.

### `GET /api/v1/experiments/compare?run_ids=run_1,run_2`
Returns aligned multi-series step-loss curves and comparative hyperparameter diff tables across selected runs.

### `POST /api/v1/experiments/rag-benchmark`
Executes empirical 4-way evaluation comparing Base Model, Base + RAG, DocTune Fine-Tuned, and Fine-Tuned + RAG across Accuracy (%), Hallucination Rate (%), Latency (ms), and Token Cost.

---

## 10. Training Queue, Sweeps & Resumption

### `GET /api/v1/training/checkpoints/resumable`
Scans output directories for valid `checkpoint-*` folders with step numbers, last loss, and epoch progress.

### `GET /api/v1/training/queue`
Returns the FIFO training job queue.

### `POST /api/v1/training/queue/cancel`
Cancels a queued job by `run_id`.

### `POST /api/v1/training/queue/clear`
Clears all queued jobs.

### `POST /api/v1/training/sweep/start`
Generates a combinatorial hyperparameter grid sweep over learning rates, LoRA ranks, and epochs, enqueuing trials sequentially into the FIFO queue.
* **Payload**:
  ```json
  {
    "model_id": "meta-llama/Meta-Llama-3-8B-Instruct",
    "method": "qlora",
    "learning_rates": [0.0001, 0.0002],
    "lora_ranks": [16, 32],
    "epochs_list": [3]
  }
  ```

---

## 11. Dataset Versioning & Provenance

### `GET /api/v1/dataset/versions`
Lists all immutable dataset snapshots with SHA-256 fingerprints, sample counts, and active status.

### `POST /api/v1/dataset/versions/activate`
Activates a historical dataset version by `version_id`, copying it to the active training dataset path.

---

## 12. Hugging Face Hub & Model Publishing

### `POST /api/v1/models/push-to-hub`
Pushes local fine-tuned adapters or merged weights to Hugging Face Hub as a public or private repository with automated Model Card generation.
* **Payload**:
  ```json
  {
    "model_name": "output_lora",
    "repo_id": "username/doctune-llama3-enterprise",
    "private": true,
    "token": "hf_..."
  }
  ```

### `POST /api/v1/models/export-gguf`
Converts merged models to GGUF format with quantization (`q4_k_m`, `q5_k_m`, `q8_0`, `f16`) and generates Ollama `Modelfile`.

### `POST /api/v1/models/ollama/register`
Registers the exported GGUF model with the local Ollama daemon.

---

## 13. Human Feedback & DPO Alignment

### `GET /api/v1/playground/dpo/stats`
Returns total counts of positive ratings, negative ratings, and recorded user corrections.

### `POST /api/v1/playground/dpo/export`
Extracts rated playground interactions into preference training JSONL pairs (`{"prompt": "...", "chosen": "...", "rejected": "..."}`).

### `POST /api/v1/playground/dpo/start`
Starts a DPO / ORPO preference alignment training run on the exported preference pairs.
