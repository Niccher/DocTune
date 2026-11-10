"""
web/schemas/job_spec.py
-----------------------
Pydantic schemas defining the unified JobSpec contract for Celery workers
and the PostgreSQL jobs queue.
"""

from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from web.state import JobType, JobStatus

JobState = JobStatus


class ResumeCheckpoint(BaseModel):
    """Details for resuming fine-tuning from an earlier checkpoint."""
    run_id: str
    step: int
    checkpoint_path: Optional[str] = None
    s3_key: Optional[str] = None


class JobSpec(BaseModel):
    """
    Complete, validated task payload dispatched to Celery workers.
    Contains all parameters required for reproducible execution.
    """
    job_id: str
    run_id: str
    display_name: str
    project_id: str = "proj_default"
    user_id: Optional[int] = None
    job_type: str = "TRAINING"                          # TRAINING, EVAL, DATA_PREP, MERGE, GGUF_EXPORT
    method: str = "sft-qlora"                           # sft-qlora, sft-lora, sft-full, dpo, orpo
    model_id: str = "meta-llama/Meta-Llama-3-8B-Instruct"

    # Dataset specification
    dataset_version_id: Optional[str] = None
    dataset_sha256: Optional[str] = None
    dataset_path: Optional[str] = None
    val_split: float = 0.1                              # 10% validation split

    # Resumption
    resume_from: Optional[ResumeCheckpoint] = None

    # Hyperparameters
    epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-4
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 1024
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: Optional[List[str]] = None          # None -> "all-linear"
    loss_on: str = "assistant"                          # "assistant" or "all"
    packing: bool = True
    early_stopping_patience: Optional[int] = 3

    # Resource requirements
    gpu_count: int = 1
    target_device: str = "auto"                         # auto, cuda, mps, cpu

    # Artifact output routing
    artifacts_root: Optional[str] = None
    output_dir: Optional[str] = None

    # Advanced options & metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
