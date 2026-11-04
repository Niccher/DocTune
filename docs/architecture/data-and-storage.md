# Data Architecture & Storage Design

DocTune Studio divides storage into **Structured Relational Metadata (PostgreSQL)** and **High-Volume Binary Object Storage (MinIO S3)**.

---

## 1. PostgreSQL Relational Entity Relationship Diagram (ERD)

```mermaid
erDiagram
    RUNS ||--o{ METRIC_POINTS : logs
    RUNS ||--o{ CHECKPOINTS : produces

    RUNS {
        string id PK "Unique UUID / Timestamp"
        string job_type "TRAINING_QLORA / TRAINING_LORA / DATA_PREP"
        string method "qlora / lora / fft"
        string model_id "Base Model ID e.g. Llama-3-8B"
        string status "PENDING / RUNNING / COMPLETED / FAILED / ABORTED"
        json config_json "Hyperparameter snapshot"
        text error_message "Traceback on failure"
        datetime created_at
        datetime started_at
        datetime completed_at
    }

    METRIC_POINTS {
        int id PK "Autoincrement"
        string run_id FK "References runs.id"
        int step "Training step counter"
        float epoch "Current epoch progress"
        float loss "Loss value"
        float eval_loss "Validation loss"
        float learning_rate "Current LR"
        datetime created_at
    }

    CHECKPOINTS {
        string id PK "Unique Checkpoint ID"
        string run_id FK "References runs.id"
        string name "Descriptive name"
        string adapter_type "qlora / lora / merged"
        string s3_bucket "S3 bucket name"
        string s3_key "Object key inside bucket"
        float size_mb "Size in Megabytes"
        float bleu_score "BLEU score"
        float rouge_score "ROUGE-L score"
        float perplexity "Perplexity metric"
        datetime created_at
    }

    UPLOADED_DOCUMENTS {
        string id PK "Document UUID"
        string filename "Original file name"
        string s3_bucket "doc-tune-raw-docs"
        string s3_key "raw_docs/uuid.pdf"
        int size_bytes "File size"
        int chunk_count "Generated chunk count"
        datetime uploaded_at
    }
```

---

## 2. MinIO S3 Bucket Hierarchy

```
MinIO Object Storage/
├── doc-tune-raw-docs/
│   ├── raw_documents/company_policy_v2.pdf
│   └── raw_documents/financial_report_2025.docx
├── doc-tune-datasets/
│   ├── 2026-11-04_run_1/private_docs.jsonl
│   └── 2026-11-04_run_1/eval_docs.jsonl
├── doc-tune-checkpoints/
│   └── run_1730732800/
│       ├── adapter_model.safetensors    (168 MB)
│       ├── adapter_config.json          (1.2 KB)
│       ├── special_tokens_map.json      (500 B)
│       └── tokenizer.json               (9.1 MB)
└── doc-tune-merged-models/
    └── run_1730732800_merged_16bit/
        ├── model-00001-of-00004.safetensors
        ├── model.safetensors.index.json
        └── config.json
```
