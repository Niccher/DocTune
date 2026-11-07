"""add users table for authentication and access control

Revision ID: 0005_add_users_and_auth
Revises: 0004_add_audit_logs
Create Date: 2026-11-07 21:50:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "0005_add_users_and_auth"
down_revision: Union[str, None] = "0004_add_audit_logs"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_tables = set(inspector.get_table_names())

    if "users" not in existing_tables:
        op.create_table(
            "users",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("email", sa.String(length=256), nullable=False),
            sa.Column("username", sa.String(length=128), nullable=False),
            sa.Column("full_name", sa.String(length=128), nullable=True),
            sa.Column("hashed_password", sa.String(length=256), nullable=False),
            sa.Column("role", sa.String(length=32), server_default="operator", nullable=False),
            sa.Column("is_active", sa.Boolean(), server_default=sa.true(), nullable=False),
            sa.Column("reset_token", sa.String(length=128), nullable=True),
            sa.Column("reset_token_expires_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        )
        op.create_index("ix_users_email", "users", ["email"], unique=True)
        op.create_index("ix_users_username", "users", ["username"], unique=True)
        op.create_index("ix_users_reset_token", "users", ["reset_token"], unique=False)
    else:
        existing_cols = {c["name"] for c in inspector.get_columns("users")}
        new_cols = [
            ("reset_token", sa.Column("reset_token", sa.String(length=128), nullable=True)),
            ("reset_token_expires_at", sa.Column("reset_token_expires_at", sa.DateTime(timezone=True), nullable=True)),
            ("last_login_at", sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=True)),
        ]
        for col_name, col_def in new_cols:
            if col_name not in existing_cols:
                op.add_column("users", col_def)


def downgrade() -> None:
    op.drop_table("users")
