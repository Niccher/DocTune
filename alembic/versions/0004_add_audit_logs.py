"""add audit logs table for operational compliance and activity tracking

Revision ID: 0004_add_audit_logs
Revises: 0003_dataset_versions_and_bigint
Create Date: 2026-11-07 20:47:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "0004_add_audit_logs"
down_revision: Union[str, None] = "0003_dataset_versions_and_bigint"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_tables = set(inspector.get_table_names())

    if "audit_logs" not in existing_tables:
        op.create_table(
            "audit_logs",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
            sa.Column("actor_ip", sa.String(length=64), nullable=True),
            sa.Column("actor_name", sa.String(length=128), server_default="System/Anonymous", nullable=False),
            sa.Column("actor_user_id", sa.String(length=64), nullable=True),
            sa.Column("action", sa.String(length=64), nullable=False),
            sa.Column("resource_type", sa.String(length=64), nullable=False),
            sa.Column("resource_id", sa.String(length=128), nullable=True),
            sa.Column("status", sa.String(length=16), server_default="SUCCESS", nullable=False),
            sa.Column("severity", sa.String(length=16), server_default="INFO", nullable=False),
            sa.Column("request_id", sa.String(length=64), nullable=True),
            sa.Column("request_method", sa.String(length=16), nullable=True),
            sa.Column("endpoint", sa.String(length=256), nullable=True),
            sa.Column("user_agent", sa.String(length=256), nullable=True),
            sa.Column("duration_ms", sa.Float(), nullable=True),
            sa.Column("details", sa.Text(), nullable=True),
        )
        op.create_index("ix_audit_logs_timestamp", "audit_logs", ["timestamp"], unique=False)
        op.create_index("ix_audit_logs_action", "audit_logs", ["action"], unique=False)
        op.create_index("ix_audit_logs_request_id", "audit_logs", ["request_id"], unique=False)
    else:
        existing_cols = {c["name"] for c in inspector.get_columns("audit_logs")}
        new_cols = [
            ("severity", sa.Column("severity", sa.String(length=16), server_default="INFO", nullable=False)),
            ("request_id", sa.Column("request_id", sa.String(length=64), nullable=True)),
            ("request_method", sa.Column("request_method", sa.String(length=16), nullable=True)),
            ("endpoint", sa.Column("endpoint", sa.String(length=256), nullable=True)),
            ("user_agent", sa.Column("user_agent", sa.String(length=256), nullable=True)),
            ("duration_ms", sa.Column("duration_ms", sa.Float(), nullable=True)),
        ]
        for col_name, col_def in new_cols:
            if col_name not in existing_cols:
                op.add_column("audit_logs", col_def)
        try:
            op.create_index("ix_audit_logs_request_id", "audit_logs", ["request_id"], unique=False)
        except Exception:
            pass


def downgrade() -> None:
    op.drop_table("audit_logs")
