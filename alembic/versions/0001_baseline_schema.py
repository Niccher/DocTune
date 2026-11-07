"""baseline schema

Revision ID: 0001_baseline_schema
Revises: 
Create Date: 2026-11-07 09:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "0001_baseline_schema"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_tables = set(inspector.get_table_names())

    # 1. runs
    if "runs" not in existing_tables:
        op.create_table(
            "runs",
            sa.Column("id", sa.String(length=64), primary_key=True),
            sa.Column("job_type", sa.String(length=32), nullable=False),
            sa.Column("method", sa.String(length=32), server_default="qlora"),
            sa.Column("model_id", sa.String(length=256), nullable=False),
            sa.Column("status", sa.String(length=32), server_default="PENDING"),
            sa.Column("dataset_path", sa.String(length=512), nullable=True),
            sa.Column("epochs", sa.Integer(), nullable=True),
            sa.Column("batch_size", sa.Integer(), nullable=True),
            sa.Column("learning_rate", sa.Float(), nullable=True),
            sa.Column("duration_sec", sa.Float(), nullable=True),
            sa.Column("final_loss", sa.Float(), nullable=True),
            sa.Column("pid", sa.Integer(), nullable=True),
            sa.Column("mode", sa.String(length=32), server_default="subprocess"),
            sa.Column("config_json", sa.JSON(), nullable=True),
            sa.Column("error_message", sa.Text(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        )

    # 2. metric_points
    if "metric_points" not in existing_tables:
        op.create_table(
            "metric_points",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("run_id", sa.String(length=64), sa.ForeignKey("runs.id", ondelete="CASCADE"), nullable=False),
            sa.Column("step", sa.Integer(), nullable=False),
            sa.Column("epoch", sa.Float(), server_default="0.0"),
            sa.Column("loss", sa.Float(), nullable=False),
            sa.Column("eval_loss", sa.Float(), nullable=True),
            sa.Column("learning_rate", sa.Float(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        )
        op.create_index("ix_metric_points_run_id", "metric_points", ["run_id"], unique=False)

    # 3. checkpoints
    if "checkpoints" not in existing_tables:
        op.create_table(
            "checkpoints",
            sa.Column("id", sa.String(length=64), primary_key=True),
            sa.Column("run_id", sa.String(length=64), sa.ForeignKey("runs.id", ondelete="CASCADE"), nullable=True),
            sa.Column("name", sa.String(length=128), nullable=False),
            sa.Column("adapter_type", sa.String(length=32), server_default="qlora"),
            sa.Column("base_model", sa.String(length=256), nullable=True),
            sa.Column("disk_path", sa.String(length=512), nullable=True),
            sa.Column("s3_bucket", sa.String(length=128), nullable=True),
            sa.Column("s3_key", sa.String(length=512), nullable=True),
            sa.Column("size_mb", sa.Float(), server_default="0.0"),
            sa.Column("step", sa.Integer(), nullable=True),
            sa.Column("val_loss", sa.Float(), nullable=True),
            sa.Column("bleu_score", sa.Float(), nullable=True),
            sa.Column("rouge_score", sa.Float(), nullable=True),
            sa.Column("perplexity", sa.Float(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        )
        op.create_index("ix_checkpoints_run_id", "checkpoints", ["run_id"], unique=False)

    # 4. uploaded_documents
    if "uploaded_documents" not in existing_tables:
        op.create_table(
            "uploaded_documents",
            sa.Column("id", sa.String(length=64), primary_key=True),
            sa.Column("filename", sa.String(length=256), nullable=False),
            sa.Column("disk_path", sa.String(length=512), nullable=True),
            sa.Column("file_hash", sa.String(length=64), nullable=True),
            sa.Column("file_format", sa.String(length=16), nullable=True),
            sa.Column("status", sa.String(length=32), server_default="READY"),
            sa.Column("s3_bucket", sa.String(length=128), nullable=True),
            sa.Column("s3_key", sa.String(length=512), nullable=True),
            sa.Column("size_bytes", sa.Integer(), server_default="0"),
            sa.Column("chunk_count", sa.Integer(), server_default="0"),
            sa.Column("uploaded_at", sa.DateTime(timezone=True), nullable=True),
        )

    # 5. system_settings
    if "system_settings" not in existing_tables:
        op.create_table(
            "system_settings",
            sa.Column("key", sa.String(length=64), primary_key=True),
            sa.Column("value_json", sa.JSON(), nullable=False),
            sa.Column("version", sa.Integer(), server_default="1"),
            sa.Column("description", sa.String(length=256), nullable=True),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        )

    # 6. dataset_samples
    if "dataset_samples" not in existing_tables:
        op.create_table(
            "dataset_samples",
            sa.Column("id", sa.String(length=64), primary_key=True),
            sa.Column("document_id", sa.String(length=64), sa.ForeignKey("uploaded_documents.id", ondelete="SET NULL"), nullable=True),
            sa.Column("run_id", sa.String(length=64), sa.ForeignKey("runs.id", ondelete="SET NULL"), nullable=True),
            sa.Column("system_prompt", sa.Text(), nullable=True),
            sa.Column("user_instruction", sa.Text(), nullable=False),
            sa.Column("assistant_response", sa.Text(), nullable=False),
            sa.Column("teacher_model", sa.String(length=128), nullable=True),
            sa.Column("status", sa.String(length=32), server_default="APPROVED"),
            sa.Column("quality_score", sa.Float(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        )

    # 7. eval_benchmarks
    if "eval_benchmarks" not in existing_tables:
        op.create_table(
            "eval_benchmarks",
            sa.Column("id", sa.String(length=64), primary_key=True),
            sa.Column("checkpoint_id", sa.String(length=64), sa.ForeignKey("checkpoints.id", ondelete="CASCADE"), nullable=True),
            sa.Column("run_id", sa.String(length=64), sa.ForeignKey("runs.id", ondelete="SET NULL"), nullable=True),
            sa.Column("test_suite_name", sa.String(length=128), server_default="domain_eval_v1"),
            sa.Column("bleu_score", sa.Float(), nullable=True),
            sa.Column("rouge_1", sa.Float(), nullable=True),
            sa.Column("rouge_2", sa.Float(), nullable=True),
            sa.Column("rouge_l", sa.Float(), nullable=True),
            sa.Column("perplexity", sa.Float(), nullable=True),
            sa.Column("avg_latency_ms", sa.Float(), nullable=True),
            sa.Column("detailed_results", sa.JSON(), nullable=True),
            sa.Column("evaluated_at", sa.DateTime(timezone=True), nullable=True),
        )

    # 8. playground_logs
    if "playground_logs" not in existing_tables:
        op.create_table(
            "playground_logs",
            sa.Column("id", sa.String(length=64), primary_key=True),
            sa.Column("prompt", sa.Text(), nullable=False),
            sa.Column("base_model_response", sa.Text(), nullable=True),
            sa.Column("tuned_model_response", sa.Text(), nullable=True),
            sa.Column("checkpoint_id", sa.String(length=64), sa.ForeignKey("checkpoints.id", ondelete="SET NULL"), nullable=True),
            sa.Column("model_name", sa.String(length=128), nullable=True),
            sa.Column("rating", sa.String(length=16), nullable=True),
            sa.Column("notes", sa.Text(), nullable=True),
            sa.Column("latency_ms", sa.Float(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        )


def downgrade() -> None:
    op.drop_table("playground_logs")
    op.drop_table("eval_benchmarks")
    op.drop_table("dataset_samples")
    op.drop_table("system_settings")
    op.drop_table("uploaded_documents")
    op.drop_table("checkpoints")
    op.drop_table("metric_points")
    op.drop_table("runs")
