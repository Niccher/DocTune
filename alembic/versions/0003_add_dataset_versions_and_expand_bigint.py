"""add dataset versions table and ensure bigint columns

Revision ID: 0003_dataset_versions_and_bigint
Revises: 0002_add_indexes
Create Date: 2026-11-07 20:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "0003_dataset_versions_and_bigint"
down_revision: Union[str, None] = "0002_add_indexes"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_tables = set(inspector.get_table_names())

    def column_exists(table_name: str, column_name: str) -> bool:
        if not inspector.has_table(table_name):
            return False
        columns = [c["name"] for c in inspector.get_columns(table_name)]
        return column_name in columns

    def ensure_column(table_name: str, column_name: str, col_type, **kw):
        if inspector.has_table(table_name) and not column_exists(table_name, column_name):
            op.add_column(table_name, sa.Column(column_name, col_type, **kw))

    # 1. dataset_versions table
    if "dataset_versions" not in existing_tables:
        op.create_table(
            "dataset_versions",
            sa.Column("id", sa.String(length=64), primary_key=True),
            sa.Column("version_tag", sa.String(length=32), nullable=False),
            sa.Column("file_path", sa.String(length=512), nullable=False),
            sa.Column("file_hash", sa.String(length=64), nullable=False),
            sa.Column("size_bytes", sa.BigInteger(), server_default="0"),
            sa.Column("sample_count", sa.Integer(), server_default="0"),
            sa.Column("description", sa.Text(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        )
        op.create_index("ix_dataset_versions_file_hash", "dataset_versions", ["file_hash"], unique=False)

    # 2. Lineage tracking on runs and checkpoints
    ensure_column("runs", "dataset_version", sa.String(length=64), nullable=True)
    ensure_column("runs", "dataset_hash", sa.String(length=64), nullable=True)
    ensure_column("checkpoints", "dataset_version", sa.String(length=64), nullable=True)
    ensure_column("checkpoints", "dataset_hash", sa.String(length=64), nullable=True)

    # 3. uploaded_documents columns safety
    ensure_column("uploaded_documents", "disk_path", sa.String(length=512), nullable=True)
    ensure_column("uploaded_documents", "file_hash", sa.String(length=64), nullable=True)
    ensure_column("uploaded_documents", "file_format", sa.String(length=16), nullable=True)
    ensure_column("uploaded_documents", "status", sa.String(length=32), server_default="READY")
    ensure_column("uploaded_documents", "s3_bucket", sa.String(length=128), nullable=True)
    ensure_column("uploaded_documents", "s3_key", sa.String(length=512), nullable=True)
    ensure_column("uploaded_documents", "chunk_count", sa.Integer(), server_default="0")


def downgrade() -> None:
    pass
