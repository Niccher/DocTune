# MinIO S3 Object Storage Service

MinIO provides high-performance, S3-compatible binary storage for uploaded raw documents, tokenized JSONL training datasets, and multi-gigabyte model checkpoint files.

---

## 1. Web Console & Access Credentials

* **Web Console URL**: `http://localhost:9001`
* **Default Username**: `minioadmin`
* **Default Password**: `minioadmin_secure_password_2026` (configurable in `.env`)
* **API S3 Endpoint**: `http://localhost:9000`

---

## 2. Bucket Layout & Access Policies

| Bucket Name | Access Policy | Contents |
| :--- | :--- | :--- |
| `doc-tune-raw-docs` | Private | Uploaded PDF, DOCX, CSV source files |
| `doc-tune-datasets` | Private | Chunked Q&A JSONL datasets |
| `doc-tune-checkpoints` | Private (Presigned URLs) | LoRA/QLoRA adapter weights (`adapter_model.safetensors`) |
| `doc-tune-merged-models`| Private (Presigned URLs) | Merged standalone 16-bit model weights |
