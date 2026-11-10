"""create datasets table and expand dataset_versions with s3_key and statistics

Revision ID: 0008_datasets_and_expanded_versions
Revises: 0007_create_jobs_table
Create Date: 2026-11-10 10:30:00.000000

"""
from typing import Sequence, Union
from datetime import datetime, timezone

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "0008_datasets_and_expanded_versions"
down_revision: Union[str, None] = "0007_create_jobs_table"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_tables = set(inspector.get_table_names())

    def column_exists(table_name: str, column_name: str) -> bool:
        if table_name not in existing_tables:
            return False
        columns = [c["name"] for c in inspector.get_columns(table_name)]
        return column_name in columns

    def ensure_column(table_name: str, column_name: str, col_type, **kw):
        if table_name in existing_tables and not column_exists(table_name, column_name):
            op.add_column(table_name, sa.Column(column_name, col_type, **kw))

    # 1. datasets
    if "datasets" not in existing_tables:
        op.create_table(
            "datasets",
            sa.Column("id", sa.String(length=64), primary_key=True),
            sa.Column("project_id", sa.String(length=64), sa.ForeignKey("projects.id", ondelete="CASCADE"), server_default="proj_default", nullable=False),
            sa.Column("name", sa.String(length=128), nullable=False),
            sa.Column("description", sa.Text(), nullable=True),
            sa.Column("source_format", sa.String(length=32), server_default="jsonl", nullable=False),
            sa.Column("created_by", sa.Integer(), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        )
        op.create_index("ix_datasets_project_id", "datasets", ["project_id"])
        existing_tables.add("datasets")

    # 2. Add columns to dataset_versions
    ensure_column("dataset_versions", "dataset_id", sa.String(length=64), nullable=True)
    ensure_column("dataset_versions", "s3_key", sa.String(length=512), nullable=True)
    ensure_column("dataset_versions", "split_type", sa.String(length=32), server_default="train", nullable=True)
    ensure_column("dataset_versions", "stats_json", sa.JSON(), nullable=True)


def downgrade() -> None:
    op.drop_table("datasets")
