import time
import os
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from pipeline.common import PROJECT_ROOT
from web.routers import dashboard, data_prep, datasets, training, models, playground, admin, inference, experiments, auth, landing
from web.services.process_manager import process_mgr
from web.db.session import init_db
from web.services.storage_service import storage_service
from web.services.redis_subscriber import start_redis_subscriber, stop_redis_subscriber
from web.services.config_service import config_service
from web.services.metrics_service import metrics_service
from web.services.auth_service import auth_service
from web.state import state


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern lifespan context manager managing application startup and graceful shutdown."""
    # ── Startup Phase ──────────────────────────────────────────────────────
    auth_service.validate_startup_security()
    init_db()
    auth_service.get_or_create_default_project()
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
    version=config_service.get_app_version(),
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)


@app.middleware("http")
async def auth_session_middleware(request: Request, call_next):
    """Resolve active user session, project context, and enforce auth/CSRF guards."""
    request.state.user = None
    request.state.project_id = "proj_default"

    # 1. Resolve session token
    token = request.cookies.get("doctune_session")
    is_cookie_auth = bool(token)
    if not token:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:].strip()

    if token:
        user_id = auth_service.verify_session_token(token)
        if user_id:
            user = auth_service.get_user_by_id(user_id)
            if user and user.is_active:
                request.state.user = user

    # 2. Resolve active project context
    project_id = (
        request.headers.get("X-Project-Id")
        or request.query_params.get("project_id")
        or request.cookies.get("doctune_project")
        or "proj_default"
    )
    request.state.project_id = project_id

    # 3. Double-submit CSRF check for mutating cookie-authenticated requests
    if is_cookie_auth and request.method in ("POST", "PUT", "DELETE", "PATCH"):
        path = request.url.path
        is_csrf_exempt = (
            path in ("/api/v1/auth/login", "/api/v1/auth/register", "/api/v1/auth/logout", "/api/v1/auth/forgot-password", "/api/v1/auth/reset-password")
            or path.startswith("/v1/")  # External API endpoints
        )
        if not is_csrf_exempt:
            csrf_cookie = request.cookies.get("doctune_csrf")
            csrf_header = request.headers.get("X-CSRF-Token")
            # If CSRF cookie is set, header must match
            if csrf_cookie and (not csrf_header or csrf_header != csrf_cookie):
                return JSONResponse(
                    {"status": "error", "message": "Invalid or missing CSRF token. Please refresh the page."},
                    status_code=403,
                )

    # 4. Route protection guard when authentication is enabled
    auth_cfg = config_service.get_auth_config()
    if auth_cfg.get("enabled", False):
        path = request.url.path
        is_exempt = (
            path.startswith("/static")
            or path in ("/health", "/healthz", "/metrics", "/api/docs", "/api/redoc")
            or path in ("/", "/features", "/security", "/docs", "/documentation")
            or path in ("/login", "/register", "/forgot-password", "/reset-password")
            or path.startswith("/api/v1/auth")
        )
        if not is_exempt and request.state.user is None:
            accept_header = request.headers.get("accept", "")
            if path.startswith("/api/") or "application/json" in accept_header:
                return JSONResponse(
                    {"status": "error", "message": "Authentication required. Please sign in."},
                    status_code=401,
                )
            query_suffix = f"?next={path}" if path != "/" else ""
            return RedirectResponse(url=f"/login{query_suffix}", status_code=302)

    return await call_next(request)


@app.middleware("http")
async def prometheus_metrics_middleware(request: Request, call_next):
    """Telemetry middleware recording request latency and HTTP request metrics."""
    request.state.start_time = time.time()
    response = await call_next(request)
    if not request.url.path.startswith("/static"):
        metrics_service.record_http_request(request.method, response.status_code)
    return response


# Ensure static directory exists and mount
static_dir = PROJECT_ROOT / "web" / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Register modular routers
app.include_router(landing.router)
app.include_router(auth.router)
app.include_router(dashboard.router)
app.include_router(data_prep.router)
app.include_router(datasets.router)
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
        "version": config_service.get_app_version(),
        "s3_storage": storage_service.is_s3_enabled(),
    }


@app.get("/healthz")
async def healthz_check():
    """Deep readiness probe checking PostgreSQL, Redis, and llama.cpp inference engine (J-03)."""
    components = {}
    is_healthy = True

    # 1. Database check
    if SessionLocal:
        try:
            from sqlalchemy import text
            db = SessionLocal()
            db.execute(text("SELECT 1"))
            db.close()
            components["database"] = {"status": "up"}
        except Exception as e:
            components["database"] = {"status": "down", "error": str(e)}
            is_healthy = False
    else:
        components["database"] = {"status": "sqlite_fallback"}

    # 2. Redis check
    try:
        import redis
        url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        r = redis.from_url(url, socket_connect_timeout=0.5)
        r.ping()
        components["redis"] = {"status": "up"}
    except Exception as e:
        components["redis"] = {"status": "down", "note": "in_memory_fallback", "error": str(e)}

    # 3. llama.cpp inference check
    try:
        from web.services.inference.manager import inference_manager
        health = await inference_manager.get_system_health()
        components["inference"] = health
    except Exception as e:
        components["inference"] = {"status": "unavailable", "error": str(e)}

    status_code = 200 if is_healthy else 503
    return JSONResponse({
        "status": "healthy" if is_healthy else "degraded",
        "app": "DocTune Studio",
        "version": config_service.get_app_version(),
        "components": components,
    }, status_code=status_code)


@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus exposition metrics endpoint for Grafana and cluster telemetry."""
    content = metrics_service.generate_prometheus_metrics()
    return Response(content=content, media_type="text/plain; version=0.0.4; charset=utf-8")
