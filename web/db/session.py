"""
web/db/session.py
-----------------
Dual-engine database session manager supporting PostgreSQL (production/docker)
with transparent SQLite fallback (local standalone development).
"""

import os
import logging
HAS_SQLALCHEMY = False
try:
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker, Session
    from web.db.models import Base
    HAS_SQLALCHEMY = True
except ImportError:
    pass

logger = logging.getLogger("doctune.db")

# Read database URL from environment
DATABASE_URL_SYNC = os.getenv("DATABASE_URL_SYNC") or os.getenv("DATABASE_URL")

engine = None
SessionLocal = None

if HAS_SQLALCHEMY:
    if DATABASE_URL_SYNC and "asyncpg" in DATABASE_URL_SYNC:
        DATABASE_URL_SYNC = DATABASE_URL_SYNC.replace("postgresql+asyncpg://", "postgresql://")

    if not DATABASE_URL_SYNC:
        # Local fallback to SQLite
        SQLITE_PATH = PROJECT_ROOT / "config" / "studio.db"
        DATABASE_URL_SYNC = f"sqlite:///{SQLITE_PATH}"
        engine = create_engine(DATABASE_URL_SYNC, connect_args={"check_same_thread": False})
    else:
        engine = create_engine(DATABASE_URL_SYNC, pool_pre_ping=True, pool_size=10, max_overflow=20)

    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Create all database tables if they do not already exist."""
    if not HAS_SQLALCHEMY or not engine:
        logger.debug("SQLAlchemy not installed; using local file store.")
        return
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database schema initialized successfully.")
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
