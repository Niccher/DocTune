"""
web/db/session.py
-----------------
Dual-engine database session manager supporting PostgreSQL (production/docker)
with transparent SQLite fallback (local standalone development).
"""

import os
import logging
from pipeline.common import PROJECT_ROOT

HAS_SQLALCHEMY = False
try:
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker, Session
    from web.db.models import Base
    HAS_SQLALCHEMY = True
except ImportError:
    Session = None
    Base = None

logger = logging.getLogger("doctune.db")

# Read database URL from environment
DATABASE_URL_SYNC = os.getenv("DATABASE_URL_SYNC") or os.getenv("DATABASE_URL")

engine = None
SessionLocal = None
if not HAS_SQLALCHEMY:
    Session = None

if HAS_SQLALCHEMY:
    # Normalize sync PostgreSQL URL to explicit psycopg2 driver
    if DATABASE_URL_SYNC:
        if "asyncpg" in DATABASE_URL_SYNC:
            DATABASE_URL_SYNC = DATABASE_URL_SYNC.replace("postgresql+asyncpg://", "postgresql+psycopg2://")
        elif DATABASE_URL_SYNC.startswith("postgresql://"):
            DATABASE_URL_SYNC = DATABASE_URL_SYNC.replace("postgresql://", "postgresql+psycopg2://")

    try:
        if DATABASE_URL_SYNC and not DATABASE_URL_SYNC.startswith("sqlite"):
            engine = create_engine(DATABASE_URL_SYNC, pool_pre_ping=True, pool_size=10, max_overflow=20)
            SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
            logger.info("Initialized PostgreSQL database engine: %s", DATABASE_URL_SYNC.split("@")[-1])
        else:
            raise ValueError("No PostgreSQL URL configured, using local SQLite")
    except Exception as e:
        logger.warning("PostgreSQL engine initialization notice: %s. Using SQLite fallback.", e)
        SQLITE_PATH = PROJECT_ROOT / "config" / "studio.db"
        SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
        sqlite_url = f"sqlite:///{SQLITE_PATH}"
        engine = create_engine(sqlite_url, connect_args={"check_same_thread": False})
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Create or upgrade database tables to the latest schema revision."""
    if not HAS_SQLALCHEMY or not engine:
        logger.debug("SQLAlchemy not installed; using local file store.")
        return
    try:
        # 1. Attempt versioned migration with Alembic if available
        try:
            from alembic.config import Config
            from alembic import command
            alembic_ini_path = PROJECT_ROOT / "alembic.ini"
            if alembic_ini_path.exists():
                alembic_cfg = Config(str(alembic_ini_path))
                command.upgrade(alembic_cfg, "head")
                logger.info("Database schema migrated to head via Alembic.")
                return
        except ImportError:
            pass
        except Exception as ae:
            logger.warning("Alembic upgrade notice (falling back to create_all): %s", ae)

        # 2. Declarative fallback
        Base.metadata.create_all(bind=engine)
        logger.info("Database schema initialized successfully via metadata.")

        # 3. Ensure newly introduced columns exist on legacy tables
        try:
            from sqlalchemy import text
            with engine.connect() as conn:
                for table_name, col_name, col_type in [
                    ("runs", "dataset_version", "VARCHAR(64)"),
                    ("runs", "dataset_hash", "VARCHAR(64)"),
                    ("checkpoints", "dataset_version", "VARCHAR(64)"),
                    ("checkpoints", "dataset_hash", "VARCHAR(64)"),
                ]:
                    try:
                        conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {col_name} {col_type}"))
                        conn.commit()
                    except Exception:
                        pass
        except Exception:
            pass
    except Exception as e:
        logger.warning("Database initialization notice (will retry on connect): %s", e)


def get_db():
    """Dependency generator yielding an active database session."""
    if not HAS_SQLALCHEMY or not SessionLocal:
        yield None
        return
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def resolve_session(db=None):
    """
    Safely resolve a database session whether called within FastAPI (with injected Session)
    or directly as a Python function (where db defaults to a Depends instance or None).
    Returns a tuple of (session, should_close).
    """
    if HAS_SQLALCHEMY and Session and isinstance(db, Session):
        return db, False
    if HAS_SQLALCHEMY and SessionLocal:
        return SessionLocal(), True
    return None, False
