"""
web/db/models.py
----------------
SQLAlchemy ORM models for runs, metrics history, checkpoints, and documents.
"""

from datetime import datetime, timezone

def now_utc():
    """Return timezone-aware current UTC datetime."""
    return datetime.now(timezone.utc)

HAS_SQLALCHEMY = False
try:
    from sqlalchemy import Column, Integer, String, Float, DateTime, Text, ForeignKey, JSON
    from sqlalchemy.orm import declarative_base, relationship
    Base = declarative_base()
    HAS_SQLALCHEMY = True
except ImportError:
    class _DummyCol:
        def __init__(self, *args, **kwargs):
            pass
    Column = Integer = String = Float = DateTime = Text = ForeignKey = JSON = _DummyCol
    def relationship(*args, **kwargs):
        return None
    Base = object



class Run(Base):
    """Tracks a training or data preparation job execution."""
    __tablename__ = "runs"

    id = Column(String(64), primary_key=True)
    job_type = Column(String(32), nullable=False)           # TRAINING_QLORA, TRAINING_LORA, DATA_PREP, MERGE
    method = Column(String(32), default="qlora")            # qlora, lora, fft
    model_id = Column(String(256), nullable=False)
    status = Column(String(32), default="PENDING", index=True)  # PENDING, RUNNING, COMPLETED, FAILED, ABORTED
    dataset_path = Column(String(512), nullable=True)
    epochs = Column(Integer, nullable=True)
    batch_size = Column(Integer, nullable=True)
    learning_rate = Column(Float, nullable=True)
    duration_sec = Column(Float, nullable=True)
    final_loss = Column(Float, nullable=True)
    pid = Column(Integer, nullable=True)
    mode = Column(String(32), default="subprocess")         # celery, subprocess
    config_json = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=now_utc, index=True)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)

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
    run_id = Column(String(64), ForeignKey("runs.id"), nullable=True, index=True)
    name = Column(String(128), nullable=False)
    adapter_type = Column(String(32), default="qlora", index=True)       # qlora, lora, merged, fft
    base_model = Column(String(256), nullable=True)
    disk_path = Column(String(512), nullable=True)
    s3_bucket = Column(String(128), nullable=True)
    s3_key = Column(String(512), nullable=True)
    size_mb = Column(Float, default=0.0)
    step = Column(Integer, nullable=True)
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
    filename = Column(String(256), nullable=False)
    disk_path = Column(String(512), nullable=True)
    file_hash = Column(String(64), nullable=True, index=True)            # SHA-256 for deduplication
    file_format = Column(String(16), nullable=True)          # pdf, docx, csv
    status = Column(String(32), default="READY")             # READY, PROCESSED, ERROR
    s3_bucket = Column(String(128), nullable=True)
    s3_key = Column(String(512), nullable=True)
    size_bytes = Column(Integer, default=0)
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
    prompt = Column(Text, nullable=False)
    base_model_response = Column(Text, nullable=True)
    tuned_model_response = Column(Text, nullable=True)
    checkpoint_id = Column(String(64), ForeignKey("checkpoints.id", ondelete="SET NULL"), nullable=True)
    model_name = Column(String(128), nullable=True)
    rating = Column(String(16), nullable=True)               # "thumbs_up", "thumbs_down"
    notes = Column(Text, nullable=True)
    latency_ms = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), default=now_utc)
