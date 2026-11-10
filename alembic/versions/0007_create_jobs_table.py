"""create jobs table for database-backed queue orchestration and heartbeat tracking

Revision ID: 0007_create_jobs_table
Revises: 0006_multi_tenancy_and_sessions
Create Date: 2026-11-10 10:00:00.000000

"""
from typing import Sequence, Union
from datetime import datetime, timezone

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "0007_create_jobs_table"
down_revision: Union[str, None] = "0006_multi_tenancy_and_sessions"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_tables = set(inspector.get_table_names())

    if "jobs" not in existing_tables:
        op.create_table(
            "jobs",
            sa.Column("id", sa.String(length=64), primary_key=True),
            sa.Column("project_id", sa.String(length=64), sa.ForeignKey("projects.id", ondelete="CASCADE"), server_default="proj_default", nullable=False),
            sa.Column("owner_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True),
            sa.Column("run_id", sa.String(length=64), nullable=True),
            sa.Column("type", sa.String(length=32), nullable=False),
            sa.Column("status", sa.String(length=32), server_default="QUEUED", nullable=False),
            sa.Column("priority", sa.Integer(), server_default="10", nullable=False),
            sa.Column("spec_json", sa.JSON(), nullable=True),
            sa.Column("gpu_req", sa.Integer(), server_default="1", nullable=False),
            sa.Column("worker_id", sa.String(length=128), nullable=True),
            sa.Column("celery_task_id", sa.String(length=128), nullable=True),
            sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("queued_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("error_code", sa.String(length=64), nullable=True),
            sa.Column("error_message", sa.Text(), nullable=True),
            sa.Column("error_details", sa.JSON(), nullable=True),
        )
        op.create_index("ix_jobs_project_id", "jobs", ["project_id"])
        op.create_index("ix_jobs_status", "jobs", ["status"])
        op.create_index("ix_jobs_run_id", "jobs", ["run_id"])
        op.create_index("ix_jobs_celery_task_id", "jobs", ["celery_task_id"])


def downgrade() -> None:
    op.drop_table("jobs")
