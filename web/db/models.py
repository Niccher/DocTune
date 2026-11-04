"""
web/db/models.py
----------------
SQLAlchemy ORM models for runs, metrics history, checkpoints, and documents.
"""

from datetime import datetime
HAS_SQLALCHEMY = False
try:
    from sqlalchemy import Column, Integer, String, Float, DateTime, Text, ForeignKey, JSON
    from sqlalchemy.orm import declarative_base, relationship
    Base = declarative_base()
    HAS_SQLALCHEMY = True
except ImportError:
    Base = object


class Run(Base):
    """Tracks a training or data preparation job execution."""
    __tablename__ = "runs"

    id = Column(String(64), primary_key=True)
    job_type = Column(String(32), nullable=False)           # TRAINING_QLORA, TRAINING_LORA, DATA_PREP, MERGE
    method = Column(String(32), default="qlora")            # qlora, lora, fft
    model_id = Column(String(256), nullable=False)
    status = Column(String(32), default="PENDING")          # PENDING, RUNNING, COMPLETED, FAILED, ABORTED
    config_json = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

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
    created_at = Column(DateTime, default=datetime.utcnow)

    run = relationship("Run", back_populates="metrics")


class Checkpoint(Base):
    """Registry of fine-tuned adapter weights and merged models stored in MinIO."""
    __tablename__ = "checkpoints"

    id = Column(String(64), primary_key=True)
    run_id = Column(String(64), ForeignKey("runs.id"), nullable=True, index=True)
    name = Column(String(128), nullable=False)
    adapter_type = Column(String(32), default="qlora")       # qlora, lora, merged, fft
    s3_bucket = Column(String(128), nullable=False)
    s3_key = Column(String(512), nullable=False)
    size_mb = Column(Float, default=0.0)
    bleu_score = Column(Float, nullable=True)
    rouge_score = Column(Float, nullable=True)
    perplexity = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    run = relationship("Run", back_populates="checkpoints")


class UploadedDocument(Base):
    """Raw uploaded enterprise documents stored in MinIO."""
    __tablename__ = "uploaded_documents"

    id = Column(String(64), primary_key=True)
    filename = Column(String(256), nullable=False)
    s3_bucket = Column(String(128), nullable=False)
    s3_key = Column(String(512), nullable=False)
    size_bytes = Column(Integer, default=0)
    chunk_count = Column(Integer, default=0)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
