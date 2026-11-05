# Security & Credential Isolation

Security practices and guidelines for handling tokens, enterprise documents, and model weights.

---

## 1. Secrets Management

* **Hugging Face User Token (`HF_TOKEN`)**:
  * Never commit `.env` or files containing tokens to version control.
  * In the UI, token inputs use `type="password"`.
  * Tokens are tested via `/api/v1/admin/test-hf` using direct HTTPS requests to Hugging Face and are never echoed back in plaintext API responses.

## 2. Document & Checkpoint Isolation

* Proprietary uploaded enterprise documents are never transmitted to third-party public cloud endpoints unless the operator explicitly configures a remote teacher LLM in `data-prep`.
* Checkpoints stored in MinIO are served through short-lived presigned URLs.
