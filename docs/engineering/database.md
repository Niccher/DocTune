# Database Architecture & Management

DocTune Studio supports dual-mode database operation: **PostgreSQL 16** for containerized/production environments and **SQLite** for zero-dependency local testing.

---

## 1. Engine & Pooling Configuration

* **Production (PostgreSQL)**: Connected via `psycopg2` (sync) and `asyncpg` (async). Utilizes connection pooling with `pool_size=10`, `max_overflow=20`, and `pool_pre_ping=True` to eliminate stale connections.
* **Development (SQLite)**: If `DATABASE_URL` is unset, the application automatically mounts `config/studio.db` using SQLite with `check_same_thread=False`.

---

## 2. Initializing Schema

Tables are defined declaratively using SQLAlchemy Base:

```bash
# Initialize database tables via Python shell
python -c "from web.db.session import init_db; init_db()"
```
