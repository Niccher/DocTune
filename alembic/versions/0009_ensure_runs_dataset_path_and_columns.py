"""ensure runs table has dataset_path and all required columns

Revision ID: 0009_ensure_runs_columns
Revises: 0008_datasets_and_versions
Create Date: 2026-11-15 09:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "0009_ensure_runs_columns"
down_revision: Union[str, None] = "0008_datasets_and_versions"
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

    # Ensure dataset_path and all required columns exist on runs
    ensure_column("runs", "dataset_path", sa.String(length=512), nullable=True)
    ensure_column("runs", "dataset_version", sa.String(length=64), nullable=True)
    ensure_column("runs", "dataset_hash", sa.String(length=64), nullable=True)
    ensure_column("runs", "manifest_json", sa.JSON(), nullable=True)
    ensure_column("runs", "config_json", sa.JSON(), nullable=True)
    ensure_column("runs", "mode", sa.String(length=32), server_default="subprocess", nullable=True)
    ensure_column("runs", "epochs", sa.Integer(), nullable=True)
    ensure_column("runs", "batch_size", sa.Integer(), nullable=True)
    ensure_column("runs", "learning_rate", sa.Float(), nullable=True)
    ensure_column("runs", "duration_sec", sa.Float(), nullable=True)
    ensure_column("runs", "final_loss", sa.Float(), nullable=True)
    ensure_column("runs", "pid", sa.Integer(), nullable=True)
    ensure_column("runs", "error_message", sa.Text(), nullable=True)
    ensure_column("runs", "created_at", sa.DateTime(timezone=True), nullable=True)
    ensure_column("runs", "started_at", sa.DateTime(timezone=True), nullable=True)
    ensure_column("runs", "completed_at", sa.DateTime(timezone=True), nullable=True)


def downgrade() -> None:
    pass
