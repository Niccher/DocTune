"""add indexes and timezone support

Revision ID: 0002_add_indexes
Revises: 0001_baseline_schema
Create Date: 2026-11-07 09:10:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "0002_add_indexes"
down_revision: Union[str, None] = "0001_baseline_schema"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    def column_exists(table_name: str, column_name: str) -> bool:
        if not inspector.has_table(table_name):
            return False
        columns = [c["name"] for c in inspector.get_columns(table_name)]
        return column_name in columns

    def ensure_column(table_name: str, column_name: str, col_type, **kw):
        if inspector.has_table(table_name) and not column_exists(table_name, column_name):
            op.add_column(table_name, sa.Column(column_name, col_type, **kw))

    def index_exists(table_name: str, index_name: str) -> bool:
        if not inspector.has_table(table_name):
            return False
        indices = [i["name"] for i in inspector.get_indexes(table_name)]
        return index_name in indices

    # Ensure required columns exist on legacy tables before indexing
    ensure_column("uploaded_documents", "file_hash", sa.String(length=64), nullable=True)
    ensure_column("uploaded_documents", "file_format", sa.String(length=16), nullable=True)
    ensure_column("uploaded_documents", "s3_bucket", sa.String(length=128), nullable=True)
    ensure_column("uploaded_documents", "s3_key", sa.String(length=512), nullable=True)
    ensure_column("uploaded_documents", "size_bytes", sa.Integer(), server_default="0")
    ensure_column("uploaded_documents", "chunk_count", sa.Integer(), server_default="0")
    ensure_column("uploaded_documents", "status", sa.String(length=32), server_default="READY")

    ensure_column("dataset_samples", "status", sa.String(length=32), server_default="APPROVED")
    ensure_column("dataset_samples", "document_id", sa.String(length=64), nullable=True)

    ensure_column("runs", "status", sa.String(length=32), server_default="PENDING")
    ensure_column("runs", "created_at", sa.DateTime(timezone=True), nullable=True)

    ensure_column("checkpoints", "adapter_type", sa.String(length=32), server_default="qlora")
    ensure_column("checkpoints", "created_at", sa.DateTime(timezone=True), nullable=True)

    # 1. uploaded_documents.file_hash
    if not index_exists("uploaded_documents", "ix_uploaded_documents_file_hash"):
        op.create_index("ix_uploaded_documents_file_hash", "uploaded_documents", ["file_hash"], unique=False)

    # 2. dataset_samples.status & document_id
    if not index_exists("dataset_samples", "ix_dataset_samples_status"):
        op.create_index("ix_dataset_samples_status", "dataset_samples", ["status"], unique=False)
    if not index_exists("dataset_samples", "ix_dataset_samples_document_id"):
        op.create_index("ix_dataset_samples_document_id", "dataset_samples", ["document_id"], unique=False)

    # 3. runs.status & created_at
    if not index_exists("runs", "ix_runs_status"):
        op.create_index("ix_runs_status", "runs", ["status"], unique=False)
    if not index_exists("runs", "ix_runs_created_at"):
        op.create_index("ix_runs_created_at", "runs", ["created_at"], unique=False)

    # 4. checkpoints.created_at & adapter_type
    if not index_exists("checkpoints", "ix_checkpoints_created_at"):
        op.create_index("ix_checkpoints_created_at", "checkpoints", ["created_at"], unique=False)
    if not index_exists("checkpoints", "ix_checkpoints_adapter_type"):
        op.create_index("ix_checkpoints_adapter_type", "checkpoints", ["adapter_type"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_checkpoints_adapter_type", table_name="checkpoints")
    op.drop_index("ix_checkpoints_created_at", table_name="checkpoints")
    op.drop_index("ix_runs_created_at", table_name="runs")
    op.drop_index("ix_runs_status", table_name="runs")
    op.drop_index("ix_dataset_samples_document_id", table_name="dataset_samples")
    op.drop_index("ix_dataset_samples_status", table_name="dataset_samples")
    op.drop_index("ix_uploaded_documents_file_hash", table_name="uploaded_documents")
