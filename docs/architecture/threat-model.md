# Threat Model & Security Architecture

DocTune Studio follows the **STRIDE** methodology to identify and mitigate potential security risks across the training lifecycle.

---

## 1. STRIDE Threat Analysis Matrix

| Threat Category | Potential Attack Vector | System Defense & Mitigation |
| :--- | :--- | :--- |
| **Spoofing** | Unauthorized user injecting training jobs via API | Internal private Docker bridge network (`doctune-net`); API rate limiting; CORS restriction on FastAPI endpoints |
| **Tampering** | Checkpoint weights corrupted in transit or replaced | MinIO S3 HMAC v4 signature verification on all uploaded `.safetensors` files; SHA-256 weight hash validation |
| **Repudiation** | Operator deleting run history or untracked changes | Immutable PostgreSQL append-only `metric_points` log; audit log tracking IP and timestamps for every run initiation |
| **Information Disclosure** | Proprietary document contents leaked through inference playground | Documents and JSONL files stored in isolated S3 buckets with time-limited (1-hour) presigned download URLs; no raw text exposed over unauthenticated public endpoints |
| **Denial of Service** | Rogue request submitting 100,000 epochs or 256 batch size freezing host | Multi-Tier Boundary Clamping in `ConfigService` and `/api/training/start` constraining CPU, RAM, and batch sizes to safe physical ceilings |
| **Elevation of Privilege** | Container breakout via PyTorch GPU drivers | Container processes run as non-root users (`doctune`); read-only filesystem mounts where appropriate; no host root volume mounts |

---

## 2. Credential Isolation

* Hugging Face user tokens (`HF_TOKEN`) and database passwords are injected strictly through `.env` or container environment variables.
* Tokens are masked in UI logs and never committed to version control (`.gitignore` enforces exclusion of `.env` and `*.key`).
