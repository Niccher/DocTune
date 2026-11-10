"""
web/db/models.py
----------------
SQLAlchemy ORM models for multi-tenant projects, users, sessions, runs,
metrics history, checkpoints, datasets, documents, and API access keys.
"""

from datetime import datetime, timezone
from typing import Optional

def now_utc():
    """Return timezone-aware current UTC datetime."""
    return datetime.now(timezone.utc)

HAS_SQLALCHEMY = False
try:
    from sqlalchemy import Column, Integer, BigInteger, String, Float, DateTime, Text, ForeignKey, JSON, Boolean
    from sqlalchemy.orm import declarative_base, relationship
    Base = declarative_base()
    HAS_SQLALCHEMY = True
except ImportError:
    class _DummyCol:
        def __init__(self, *args, **kwargs):
            pass
    Column = Integer = BigInteger = String = Float = DateTime = Text = ForeignKey = JSON = Boolean = _DummyCol
    def relationship(*args, **kwargs):
        return None
    Base = object


# ── Multi-Tenancy & Project Models ───────────────────────────────────────────

class Project(Base):
    """Isolated project / workspace boundary for multi-tenant resource management."""
    __tablename__ = "projects"

    id = Column(String(64), primary_key=True)               # e.g., proj_default or proj_01HP...
    name = Column(String(128), nullable=False)
    slug = Column(String(64), unique=True, index=True, nullable=False)
    description = Column(Text, nullable=True)
    storage_quota_bytes = Column(BigInteger, default=53687091200, nullable=False)  # default 50 GB
    created_by = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    created_at = Column(DateTime(timezone=True), default=now_utc, nullable=False)

    members = relationship("ProjectMember", back_populates="project", cascade="all, delete-orphan")
    runs = relationship("Run", back_populates="project", cascade="all, delete-orphan")
    jobs = relationship("Job", back_populates="project", cascade="all, delete-orphan")
    datasets = relationship("Dataset", back_populates="project", cascade="all, delete-orphan")
    api_keys = relationship("ApiKey", back_populates="project", cascade="all, delete-orphan")
    provider_credentials = relationship("ProviderCredential", back_populates="project", cascade="all, delete-orphan")


class Job(Base):
    """Transactional queue record for Celery background tasks and worker heartbeats."""
    __tablename__ = "jobs"

    id = Column(String(64), primary_key=True)               # e.g., job_train_01HP...
    project_id = Column(String(64), ForeignKey("projects.id", ondelete="CASCADE"), default="proj_default", nullable=False, index=True)
    owner_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    run_id = Column(String(64), nullable=True, index=True)
    type = Column(String(32), nullable=False)               # TRAINING, EVAL, DATA_PREP, MERGE, GGUF_EXPORT, DEPLOY
    status = Column(String(32), default="QUEUED", index=True, nullable=False) # QUEUED, ASSIGNED, RUNNING, SUCCEEDED, FAILED, CANCELLED
    priority = Column(Integer, default=10, nullable=False)
    spec_json = Column(JSON, nullable=True)
    gpu_req = Column(Integer, default=1, nullable=False)
    worker_id = Column(String(128), nullable=True)
    celery_task_id = Column(String(128), nullable=True, index=True)
    heartbeat_at = Column(DateTime(timezone=True), nullable=True)
    queued_at = Column(DateTime(timezone=True), default=now_utc, nullable=False)
    started_at = Column(DateTime(timezone=True), nullable=True)
    finished_at = Column(DateTime(timezone=True), nullable=True)
    error_code = Column(String(64), nullable=True)
    error_message = Column(Text, nullable=True)
    error_details = Column(JSON, nullable=True)

    project = relationship("Project", back_populates="jobs")


class ProjectMember(Base):
    """Maps user memberships and role-based access per project."""
    __tablename__ = "project_members"

    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(String(64), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    role = Column(String(32), default="operator", nullable=False)  # admin, operator, viewer
    created_at = Column(DateTime(timezone=True), default=now_utc, nullable=False)

    project = relationship("Project", back_populates="members")
    user = relationship("User", back_populates="project_memberships")


class UserSession(Base):
    """Server-side session store with cryptographic token hashing and instant revocation."""
    __tablename__ = "sessions"

    id = Column(String(64), primary_key=True)               # e.g., sess_01HP...
    token_hash = Column(String(64), unique=True, index=True, nullable=False)  # SHA-256 of 32-byte secret
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    ip_address = Column(String(64), nullable=True)
    user_agent = Column(String(256), nullable=True)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    revoked_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=now_utc, nullable=False)

    user = relationship("User", back_populates="sessions")


class ApiKey(Base):
    """Programmatic API keys for external /v1/ inference and pipeline access."""
    __tablename__ = "api_keys"

    id = Column(String(64), primary_key=True)               # e.g., key_01HP...
    project_id = Column(String(64), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    name = Column(String(128), nullable=False)
    key_hash = Column(String(64), unique=True, index=True, nullable=False)   # SHA-256 of dt_live_...
    prefix = Column(String(16), nullable=False)                             # dt_live_xxxx
    role = Column(String(32), default="operator", nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    revoked_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=now_utc, nullable=False)

    @property
    def is_active(self) -> bool:
        return self.revoked_at is None

    project = relationship("Project", back_populates="api_keys")



class ProviderCredential(Base):
    """Encrypted third-party provider API keys (OpenAI, Anthropic, HuggingFace)."""
    __tablename__ = "provider_credentials"

    id = Column(String(64), primary_key=True)
    project_id = Column(String(64), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True)
    provider = Column(String(32), nullable=False)            # openai, anthropic, huggingface
    key_ciphertext = Column(Text, nullable=False)            # Fernet encrypted ciphertext
    created_at = Column(DateTime(timezone=True), default=now_utc, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=now_utc, onupdate=now_utc, nullable=False)

    project = relationship("Project", back_populates="provider_credentials")


# ── Resource Entities (Project Scoped) ───────────────────────────────────────

class Run(Base):
    """Tracks a training or data preparation job execution."""
    __tablename__ = "runs"

    id = Column(String(64), primary_key=True)
    project_id = Column(String(64), ForeignKey("projects.id", ondelete="SET NULL"), default="proj_default", nullable=True, index=True)
    created_by = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    job_type = Column(String(32), nullable=False)           # TRAINING_QLORA, TRAINING_LORA, DATA_PREP, MERGE
    method = Column(String(32), default="qlora")            # qlora, lora, fft
    model_id = Column(String(256), nullable=False)
    status = Column(String(32), default="PENDING", index=True)  # PENDING, RUNNING, COMPLETED, FAILED, ABORTED
    dataset_path = Column(String(512), nullable=True)
    dataset_version = Column(String(64), nullable=True)
    dataset_hash = Column(String(64), nullable=True)
    epochs = Column(Integer, nullable=True)
    batch_size = Column(Integer, nullable=True)
    learning_rate = Column(Float, nullable=True)
    duration_sec = Column(Float, nullable=True)
    final_loss = Column(Float, nullable=True)
    pid = Column(Integer, nullable=True)
    mode = Column(String(32), default="subprocess")         # celery, subprocess
    config_json = Column(JSON, nullable=True)
    manifest_json = Column(JSON, nullable=True)             # Reproducibility manifest (versions, hardware, config)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=now_utc, index=True)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    project = relationship("Project", back_populates="runs")
    metrics = relationship("MetricPoint", back_populates="run", cascade="all, delete-orphan")
    checkpoints = relationship("Checkpoint", back_populates="run", cascade="all, delete-orphan")


class MetricPoint(Base):
    """Step-by-step training telemetry for interactive loss curve plotting."""
    __tablename__ = "metric_points"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(64), ForeignKey("runs.id"), nullable=False, index=True)
    step = Column(Integer, nullable=False)
    epoch = Column(Float, default=0.0)
    loss = Column(Float, nullable=False)
    eval_loss = Column(Float, nullable=True)
    learning_rate = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), default=now_utc)

    run = relationship("Run", back_populates="metrics")


class Checkpoint(Base):
    """Registry of fine-tuned adapter weights and merged models stored in MinIO and disk."""
    __tablename__ = "checkpoints"

    id = Column(String(64), primary_key=True)
    project_id = Column(String(64), ForeignKey("projects.id", ondelete="SET NULL"), default="proj_default", nullable=True, index=True)
    created_by = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    run_id = Column(String(64), ForeignKey("runs.id"), nullable=True, index=True)
    name = Column(String(128), nullable=False)
    adapter_type = Column(String(32), default="qlora", index=True)       # qlora, lora, merged, fft, gguf
    base_model = Column(String(256), nullable=True)
    disk_path = Column(String(512), nullable=True)
    s3_bucket = Column(String(128), nullable=True)
    s3_key = Column(String(512), nullable=True)
    size_mb = Column(Float, default=0.0)
    step = Column(Integer, nullable=True)
    dataset_version = Column(String(64), nullable=True)
    dataset_hash = Column(String(64), nullable=True)
    val_loss = Column(Float, nullable=True)
    bleu_score = Column(Float, nullable=True)
    rouge_score = Column(Float, nullable=True)
    perplexity = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), default=now_utc, index=True)

    run = relationship("Run", back_populates="checkpoints")


class UploadedDocument(Base):
    """Raw uploaded enterprise documents stored in local storage and MinIO."""
    __tablename__ = "uploaded_documents"

    id = Column(String(64), primary_key=True)
    project_id = Column(String(64), ForeignKey("projects.id", ondelete="SET NULL"), default="proj_default", nullable=True, index=True)
    created_by = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    filename = Column(String(256), nullable=False)
    disk_path = Column(String(512), nullable=True)
    file_hash = Column(String(64), nullable=True, index=True)            # SHA-256 for deduplication
    file_format = Column(String(16), nullable=True)                      # pdf, docx, csv, txt, md
    status = Column(String(32), default="READY")                         # READY, PROCESSED, ERROR
    s3_bucket = Column(String(128), nullable=True)
    s3_key = Column(String(512), nullable=True)
    size_bytes = Column(BigInteger, default=0)
    chunk_count = Column(Integer, default=0)
    uploaded_at = Column(DateTime(timezone=True), default=now_utc)


class SystemSetting(Base):
    """Key-value structured system configuration persisted in database with JSON fallback."""
    __tablename__ = "system_settings"

    key = Column(String(64), primary_key=True)               # e.g., "server", "compute_resources", etc.
    value_json = Column(JSON, nullable=False)
    version = Column(Integer, default=1)
    description = Column(String(256), nullable=True)
    updated_at = Column(DateTime(timezone=True), default=now_utc, onupdate=now_utc)


class DatasetSample(Base):
    """Instruction tuning Q&A pair generated from document chunks with curation workflow."""
    __tablename__ = "dataset_samples"

    id = Column(String(64), primary_key=True)
    project_id = Column(String(64), ForeignKey("projects.id", ondelete="SET NULL"), default="proj_default", nullable=True, index=True)
    created_by = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    document_id = Column(String(64), ForeignKey("uploaded_documents.id", ondelete="SET NULL"), nullable=True, index=True)
    run_id = Column(String(64), ForeignKey("runs.id", ondelete="SET NULL"), nullable=True)
    system_prompt = Column(Text, nullable=True)
    user_instruction = Column(Text, nullable=False)
    assistant_response = Column(Text, nullable=False)
    teacher_model = Column(String(128), nullable=True)
    status = Column(String(32), default="APPROVED", index=True)          # APPROVED, REJECTED, EDITED
    quality_score = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), default=now_utc)


class EvaluationBenchmark(Base):
    """Automated benchmark test results evaluating models against standard domain test sets."""
    __tablename__ = "eval_benchmarks"

    id = Column(String(64), primary_key=True)
    project_id = Column(String(64), ForeignKey("projects.id", ondelete="SET NULL"), default="proj_default", nullable=True, index=True)
    created_by = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    checkpoint_id = Column(String(64), ForeignKey("checkpoints.id", ondelete="CASCADE"), nullable=True)
    run_id = Column(String(64), ForeignKey("runs.id", ondelete="SET NULL"), nullable=True)
    test_suite_name = Column(String(128), default="domain_eval_v1")
    bleu_score = Column(Float, nullable=True)
    rouge_1 = Column(Float, nullable=True)
    rouge_2 = Column(Float, nullable=True)
    rouge_l = Column(Float, nullable=True)
    perplexity = Column(Float, nullable=True)
    avg_latency_ms = Column(Float, nullable=True)
    detailed_results = Column(JSON, nullable=True)
    evaluated_at = Column(DateTime(timezone=True), default=now_utc)


class PlaygroundLog(Base):
    """Prompt comparison history and human evaluation feedback."""
    __tablename__ = "playground_logs"

    id = Column(String(64), primary_key=True)
    project_id = Column(String(64), ForeignKey("projects.id", ondelete="SET NULL"), default="proj_default", nullable=True, index=True)
    created_by = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    prompt = Column(Text, nullable=False)
    base_model_response = Column(Text, nullable=True)
    tuned_model_response = Column(Text, nullable=True)
    checkpoint_id = Column(String(64), ForeignKey("checkpoints.id", ondelete="SET NULL"), nullable=True)
    model_name = Column(String(128), nullable=True)
    rating = Column(String(16), nullable=True)               # "thumbs_up", "thumbs_down"
    notes = Column(Text, nullable=True)
    latency_ms = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), default=now_utc)


class Dataset(Base):
    """Collection of instruction tuning datasets partitioned by project."""
    __tablename__ = "datasets"

    id = Column(String(64), primary_key=True)               # e.g., ds_01HP...
    project_id = Column(String(64), ForeignKey("projects.id", ondelete="CASCADE"), default="proj_default", nullable=False, index=True)
    name = Column(String(128), nullable=False)
    description = Column(Text, nullable=True)
    source_format = Column(String(32), default="jsonl", nullable=False) # jsonl, alpaca, openai, parquet, csv, hf
    created_by = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    created_at = Column(DateTime(timezone=True), default=now_utc, nullable=False)

    project = relationship("Project", back_populates="datasets")
    versions = relationship("DatasetVersion", back_populates="dataset", cascade="all, delete-orphan")


class DatasetVersion(Base):
    """Immutable tagged snapshots of instruction tuning datasets for lineage tracking."""
    __tablename__ = "dataset_versions"

    id = Column(String(64), primary_key=True)               # e.g., ds_v1_172345678_a1b2c3d4
    dataset_id = Column(String(64), ForeignKey("datasets.id", ondelete="SET NULL"), nullable=True, index=True)
    project_id = Column(String(64), ForeignKey("projects.id", ondelete="SET NULL"), default="proj_default", nullable=True, index=True)
    created_by = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    version_tag = Column(String(32), nullable=False)        # e.g., v1, v2.1, q4-qa
    file_path = Column(String(512), nullable=False)
    s3_key = Column(String(512), nullable=True)
    file_hash = Column(String(64), nullable=False, index=True) # SHA-256
    split_type = Column(String(32), default="train", nullable=True)
    size_bytes = Column(BigInteger, default=0)
    sample_count = Column(Integer, default=0)
    stats_json = Column(JSON, nullable=True)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=now_utc)

    dataset = relationship("Dataset", back_populates="versions")


class AuditLog(Base):
    """Immutable audit trail recording state-modifying actions and actor metadata."""
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime(timezone=True), default=now_utc, nullable=False, index=True)
    actor_ip = Column(String(64), nullable=True)
    actor_name = Column(String(128), default="System/Anonymous", nullable=False)
    actor_user_id = Column(String(64), nullable=True)
    action = Column(String(64), nullable=False, index=True)
    resource_type = Column(String(64), nullable=False)
    resource_id = Column(String(128), nullable=True)
    status = Column(String(16), default="SUCCESS", nullable=False)
    severity = Column(String(16), default="INFO", nullable=False)
    request_id = Column(String(64), nullable=True, index=True)
    request_method = Column(String(16), nullable=True)
    endpoint = Column(String(256), nullable=True)
    user_agent = Column(String(256), nullable=True)
    duration_ms = Column(Float, nullable=True)
    details = Column(Text, nullable=True)


class User(Base):
    """User account model for authentication, access control, and audit tracking."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String(256), unique=True, index=True, nullable=False)
    username = Column(String(128), unique=True, index=True, nullable=False)
    full_name = Column(String(128), nullable=True)
    hashed_password = Column(String(256), nullable=False)
    role = Column(String(32), default="operator", nullable=False)  # admin, operator, viewer
    is_active = Column(Boolean, default=True, nullable=False)
    reset_token = Column(String(128), nullable=True, index=True)
    reset_token_expires_at = Column(DateTime(timezone=True), nullable=True)
    last_login_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=now_utc)

    sessions = relationship("UserSession", back_populates="user", cascade="all, delete-orphan")
    project_memberships = relationship("ProjectMember", back_populates="user", cascade="all, delete-orphan")


def scoped(query, model, project_id: Optional[str] = None):
    """Scope an SQLAlchemy query to a specific project if model supports project_id."""
    if project_id and hasattr(model, "project_id"):
        return query.filter(getattr(model, "project_id") == project_id)
    return query
