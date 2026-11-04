# REST & WebSocket API Contract Reference

DocTune Studio exposes a clean REST API and WebSocket streams for external automation, pipeline orchestration, and custom dashboard integrations.

---

## 1. REST Endpoints

### Training Studio Endpoints

#### `POST /api/training/start`
Starts a new fine-tuning run.
* **Form Parameters**:
  * `method` (string): `qlora`, `lora`, or `fft`. Default: `qlora`.
  * `model_id` (string): Hugging Face model repository ID.
  * `epochs` (integer): Number of training epochs (clamped [1, 50]).
  * `batch_size` (integer): Per-device batch size (clamped [1, 32]).
  * `grad_accum` (integer): Gradient accumulation steps (clamped [1, 64]).
  * `lora_r` (integer): LoRA attention dimension rank (clamped [2, 256]).
  * `lora_alpha` (integer): LoRA alpha scaling factor (clamped [2, 512]).
  * `learning_rate` (float): Training learning rate (clamped [1e-7, 0.1]).
  * `max_seq_length` (integer): Maximum sequence context tokens (clamped [128, 8192]).
* **Response**: `200 OK`
  ```json
  {
    "status": "started",
    "job_id": "run_1730732800"
  }
  ```

#### `POST /api/training/stop`
Aborts the currently active training or data preparation process.
* **Response**: `200 OK`
  ```json
  {
    "status": "success",
    "message": "Job abort signal sent"
  }
  ```

---

### System Administration Endpoints

#### `GET /api/admin/hardware-bounds`
Returns live physical host boundary metrics.
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

#### `POST /api/admin/settings`
Updates system configurations with automatic boundary clamping.
* **Payload**: Complete or partial settings JSON object.
* **Response**: `200 OK` with updated settings and informative clamping notices.

---

## 2. WebSocket Telemetry Streaming

### `WS /ws/stream` & `WS /ws/training`
Full-duplex WebSocket stream for real-time loss plotting and log streaming.

* **Client Message**: Send `"ping"` to receive `"pong"` heartbeat.
* **Server Frames**:
  * Log lines:
    ```json
    "Epoch 1/3: 25% [=====>     ] loss=1.423"
    ```
  * Structured metric lines:
    ```json
    "[METRIC_JSON] {\"step\": 20, \"loss\": 1.35, \"epoch\": 0.4}"
    ```
