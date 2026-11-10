"""add multi-tenancy projects, sessions, api keys, and provider credentials

Revision ID: 0006_multi_tenancy_and_sessions
Revises: 0005_add_users_and_auth
Create Date: 2026-11-10 09:30:00.000000

"""
from typing import Sequence, Union
from datetime import datetime, timezone

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "0006_multi_tenancy_and_sessions"
down_revision: Union[str, None] = "0005_add_users_and_auth"
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

    # 1. projects
    if "projects" not in existing_tables:
        op.create_table(
            "projects",
            sa.Column("id", sa.String(length=64), primary_key=True),
            sa.Column("name", sa.String(length=128), nullable=False),
            sa.Column("slug", sa.String(length=64), nullable=False),
            sa.Column("description", sa.Text(), nullable=True),
            sa.Column("storage_quota_bytes", sa.BigInteger(), server_default="53687091200", nullable=False),
            sa.Column("created_by", sa.Integer(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        )
        op.create_index("ix_projects_slug", "projects", ["slug"], unique=True)
        existing_tables.add("projects")

    # 2. project_members
    if "project_members" not in existing_tables:
        op.create_table(
            "project_members",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("project_id", sa.String(length=64), sa.ForeignKey("projects.id", ondelete="CASCADE"), nullable=False),
            sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
            sa.Column("role", sa.String(length=32), server_default="operator", nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        )
        op.create_index("ix_project_members_project_id", "project_members", ["project_id"])
        op.create_index("ix_project_members_user_id", "project_members", ["user_id"])
        existing_tables.add("project_members")

    # 3. sessions
    if "sessions" not in existing_tables:
        op.create_table(
            "sessions",
            sa.Column("id", sa.String(length=64), primary_key=True),
            sa.Column("token_hash", sa.String(length=64), nullable=False),
            sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
            sa.Column("ip_address", sa.String(length=64), nullable=True),
            sa.Column("user_agent", sa.String(length=256), nullable=True),
            sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        )
        op.create_index("ix_sessions_token_hash", "sessions", ["token_hash"], unique=True)
        op.create_index("ix_sessions_user_id", "sessions", ["user_id"])
        existing_tables.add("sessions")

    # 4. api_keys
    if "api_keys" not in existing_tables:
        op.create_table(
            "api_keys",
            sa.Column("id", sa.String(length=64), primary_key=True),
            sa.Column("project_id", sa.String(length=64), sa.ForeignKey("projects.id", ondelete="CASCADE"), nullable=False),
            sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
            sa.Column("name", sa.String(length=128), nullable=False),
            sa.Column("key_hash", sa.String(length=64), nullable=False),
            sa.Column("prefix", sa.String(length=16), nullable=False),
            sa.Column("role", sa.String(length=32), server_default="operator", nullable=False),
            sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        )
        op.create_index("ix_api_keys_key_hash", "api_keys", ["key_hash"], unique=True)
        op.create_index("ix_api_keys_project_id", "api_keys", ["project_id"])
        existing_tables.add("api_keys")

    # 5. provider_credentials
    if "provider_credentials" not in existing_tables:
        op.create_table(
            "provider_credentials",
            sa.Column("id", sa.String(length=64), primary_key=True),
            sa.Column("project_id", sa.String(length=64), sa.ForeignKey("projects.id", ondelete="CASCADE"), nullable=False),
            sa.Column("provider", sa.String(length=32), nullable=False),
            sa.Column("key_ciphertext", sa.Text(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        )
        op.create_index("ix_provider_credentials_project_id", "provider_credentials", ["project_id"])
        existing_tables.add("provider_credentials")

    # Add project_id and created_by to resource tables
    resource_tables = [
        "runs",
        "checkpoints",
        "uploaded_documents",
        "dataset_samples",
        "dataset_versions",
        "eval_benchmarks",
        "playground_logs",
    ]

    for tbl in resource_tables:
        ensure_column(tbl, "project_id", sa.String(length=64), server_default="proj_default", nullable=True)
        ensure_column(tbl, "created_by", sa.Integer(), nullable=True)

    # Backfill default project if not present
    now = datetime.now(timezone.utc)
    try:
        bind.execute(
            sa.text(
                "INSERT INTO projects (id, name, slug, description, storage_quota_bytes, created_at) "
                "VALUES ('proj_default', 'Default Workspace', 'default', 'Auto-provisioned default workspace', 53687091200, :now) "
                "ON CONFLICT (id) DO NOTHING"
            ),
            {"now": now},
        )
    except Exception:
        # SQLite or engines without ON CONFLICT syntax
        try:
            res = bind.execute(sa.text("SELECT id FROM projects WHERE id = 'proj_default'")).fetchone()
            if not res:
                bind.execute(
                    sa.text(
                        "INSERT INTO projects (id, name, slug, description, storage_quota_bytes, created_at) "
                        "VALUES ('proj_default', 'Default Workspace', 'default', 'Auto-provisioned default workspace', 53687091200, :now)"
                    ),
                    {"now": now},
                )
        except Exception:
            pass


def downgrade() -> None:
    for tbl in [
        "runs",
        "checkpoints",
        "uploaded_documents",
        "dataset_samples",
        "dataset_versions",
        "eval_benchmarks",
        "playground_logs",
    ]:
        try:
            op.drop_column(tbl, "project_id")
            op.drop_column(tbl, "created_by")
        except Exception:
            pass

    op.drop_table("provider_credentials")
    op.drop_table("api_keys")
    op.drop_table("sessions")
    op.drop_table("project_members")
    op.drop_table("projects")
