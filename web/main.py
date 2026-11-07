from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from pipeline.common import PROJECT_ROOT
from web.routers import dashboard, data_prep, training, models, playground, admin, inference, experiments
from web.services.process_manager import process_mgr
from web.db.session import init_db
from web.services.storage_service import storage_service
from web.services.redis_subscriber import start_redis_subscriber, stop_redis_subscriber
from web.state import state


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern lifespan context manager managing application startup and graceful shutdown."""
    # ── Startup Phase ──────────────────────────────────────────────────────
    init_db()
    storage_service.ensure_buckets_exist()
    state.rehydrate_from_db()
    start_redis_subscriber()

    yield

    # ── Shutdown Phase ─────────────────────────────────────────────────────
    await stop_redis_subscriber()
    await process_mgr.abort_current_job()


app = FastAPI(
    title="Private Document Fine-Tuning Studio",
    description="AdminLTE v4 dashboard for managing LLM document fine-tuning pipelines",
    version="1.0.0",
    lifespan=lifespan,
)

# Ensure static directory exists and mount
static_dir = PROJECT_ROOT / "web" / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Register modular routers
app.include_router(dashboard.router)
app.include_router(data_prep.router)
app.include_router(training.router)
app.include_router(models.router)
app.include_router(playground.router)
app.include_router(admin.router)
app.include_router(inference.router)
app.include_router(experiments.router)


@app.get("/health")
async def health_check():
    """Healthcheck probe for Docker containers and load balancers."""
    return {
        "status": "healthy",
        "app": "DocTune Studio",
        "version": "1.0.0",
        "s3_storage": storage_service.is_s3_enabled(),
    }
