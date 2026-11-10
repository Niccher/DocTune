"""
web/routers/auth.py
-------------------
Authentication endpoints for user registration, login, logout,
password recovery, project membership, and role-based access control (RBAC).
"""

import os
import secrets
import logging
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, Request, Response, HTTPException, status, Depends
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from sqlalchemy.orm import Session

from web.services.auth_service import auth_service
from web.services.audit_service import audit_service
from web.services.config_service import config_service
from web.templates_config import templates
from web.db.models import User, Project, ProjectMember
from web.db.session import get_db, resolve_session

logger = logging.getLogger("doctune.auth")
router = APIRouter(tags=["authentication"])


# ── Dependency Guards for Route Protection ───────────────────────────────────

def get_current_user(request: Request) -> Optional[User]:
    """Retrieve authenticated user from request state if present."""
    return getattr(request.state, "user", None)


def require_user(request: Request) -> User:
    """Enforce that the request is made by an active authenticated user."""
    auth_cfg = config_service.get_auth_config()
    if not auth_cfg.get("enabled", False):
        return User(id="usr_default", username="admin", email="admin@doctune.local", role="admin", is_active=True)
    user = getattr(request.state, "user", None)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please sign in.",
        )
    return user


def require_site_admin(request: Request) -> User:
    """Enforce site-wide administrator permissions."""
    user = require_user(request)
    if user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="System administrator permissions required.",
        )
    return user


def require_project_role(min_role: str = "viewer"):
    """
    Factory dependency verifying minimum required project role ('viewer', 'operator', 'admin').
    Role hierarchy: viewer (read) < operator (execute runs/evals) < admin (manage members/keys).
    """
    hierarchy = {"viewer": 1, "operator": 2, "admin": 3}
    required_level = hierarchy.get(min_role.lower(), 1)

    async def _verifier(request: Request) -> User:
        user = require_user(request)
        if user.role == "admin":
            return user  # System administrators bypass project-level limits

        project_id = (
            request.headers.get("X-Project-Id")
            or request.query_params.get("project_id")
            or request.cookies.get("doctune_project")
            or "proj_default"
        )
        user_role = auth_service.get_user_project_role(user.id, project_id)
        current_level = hierarchy.get((user_role or "").lower(), 0)

        if current_level < required_level:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions for project '{project_id}'. Requires '{min_role}' role or higher.",
            )
        return user

    return _verifier


def get_project_id(request: Request) -> str:
    """Extract resolved project ID from request state, header, cookie or query param."""
    if hasattr(request.state, "project_id") and request.state.project_id:
        return request.state.project_id
    return (
        request.headers.get("X-Project-Id")
        or request.query_params.get("project_id")
        or request.cookies.get("doctune_project")
        or "proj_default"
    )


def scoped_query(db: Any, model: Any, project_id: str):
    """
    Return a SQLAlchemy Query scoped to the given project_id if the model has a project_id column.
    Ensures cross-tenant query isolation (G-08).
    """
    query = db.query(model)
    if hasattr(model, "project_id"):
        query = query.filter(model.project_id == project_id)
    return query


# ── Redis / In-Memory Login Rate Limiting (G-07) ──────────────────────────────

_LOGIN_ATTEMPTS: Dict[str, List[float]] = {}
_REDIS_CLIENT = None
_REDIS_CHECKED = False

def _get_redis():
    global _REDIS_CLIENT, _REDIS_CHECKED
    if not _REDIS_CHECKED:
        _REDIS_CHECKED = True
        try:
            import redis
            url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
            client = redis.from_url(url, decode_responses=True, socket_connect_timeout=0.5)
            client.ping()
            _REDIS_CLIENT = client
        except Exception:
            _REDIS_CLIENT = None
    return _REDIS_CLIENT

def check_login_rate_limit(key: str, limit: int = 5, window_sec: int = 60) -> bool:
    """Rate limiter checking both Redis (shared across workers) and in-memory fallback."""
    r = _get_redis()
    if r:
        try:
            r_key = f"rate:login:{key}"
            count = r.get(r_key)
            if count is not None and int(count) >= limit:
                return False
        except Exception:
            pass

    import time
    now = time.time()
    history = _LOGIN_ATTEMPTS.setdefault(key, [])
    _LOGIN_ATTEMPTS[key] = [t for t in history if now - t < window_sec]
    return len(_LOGIN_ATTEMPTS[key]) < limit

def record_failed_attempt(key: str, window_sec: int = 60):
    """Record a failed login attempt in Redis and in-memory."""
    r = _get_redis()
    if r:
        try:
            r_key = f"rate:login:{key}"
            pipe = r.pipeline()
            pipe.incr(r_key)
            pipe.expire(r_key, window_sec)
            pipe.execute()
        except Exception:
            pass

    import time
    _LOGIN_ATTEMPTS.setdefault(key, []).append(time.time())

def clear_failed_attempts(key: str):
    """Clear failed login counter upon successful authentication."""
    r = _get_redis()
    if r:
        try:
            r.delete(f"rate:login:{key}")
        except Exception:
            pass
    _LOGIN_ATTEMPTS.pop(key, None)


# ── UI Route Handlers ─────────────────────────────────────────────────────────

def _safe_next_url(url: Optional[str]) -> str:
    """Validate and sanitize redirect URL to prevent open redirect vulnerabilities."""
    if not url or not isinstance(url, str):
        return "/dashboard"
    trimmed = url.strip()
    if not trimmed.startswith("/") or trimmed.startswith("//") or "\\" in trimmed or trimmed == "/":
        return "/dashboard"
    return trimmed


@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, next: str = "/dashboard"):
    """Render the user login page."""
    safe_next = _safe_next_url(next)
    user = getattr(request.state, "user", None)
    if user:
        return RedirectResponse(url=safe_next, status_code=status.HTTP_302_FOUND)

    user_count = auth_service.count_users()
    if user_count == 0:
        return RedirectResponse(url="/register", status_code=status.HTTP_302_FOUND)

    return templates.TemplateResponse(
        request=request,
        name="auth/login.html",
        context={
            "page_title": "Sign In — DocTune Studio",
            "next_url": safe_next,
            "app_name": config_service.get_app_name(),
            "user_count": user_count,
        },
    )


@router.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    """Render the user registration page."""
    user = getattr(request.state, "user", None)
    user_count = auth_service.count_users()
    is_first_user = (user_count == 0)

    auth_cfg = config_service.get_auth_config()
    if not is_first_user and not auth_cfg.get("allow_registration", True) and not (user and user.role == "admin"):
        return RedirectResponse(url="/login?error=Registration+is+disabled", status_code=status.HTTP_302_FOUND)

    return templates.TemplateResponse(
        request=request,
        name="auth/register.html",
        context={
            "page_title": "Register Administrator" if is_first_user else "Create Account — DocTune Studio",
            "is_first_user": is_first_user,
            "user_count": user_count,
            "app_name": config_service.get_app_name(),
        },
    )


@router.get("/forgot-password", response_class=HTMLResponse)
async def register_forgot_password_page(request: Request):
    """Render the password reset request page."""
    return templates.TemplateResponse(
        request=request,
        name="auth/forgot_password.html",
        context={
            "page_title": "Forgot Password — DocTune Studio",
            "app_name": config_service.get_app_name(),
        },
    )


@router.get("/reset-password", response_class=HTMLResponse)
async def reset_password_page(request: Request, token: str = ""):
    """Render the new password entry page."""
    return templates.TemplateResponse(
        request=request,
        name="auth/reset_password.html",
        context={
            "page_title": "Reset Password — DocTune Studio",
            "token": token,
            "app_name": config_service.get_app_name(),
        },
    )


# ── JSON & Form Action Handlers ───────────────────────────────────────────────

@router.post("/api/v1/auth/login", response_class=JSONResponse)
async def login_api(request: Request):
    """Authenticate user credentials, start server-side session, set secure cookies."""
    client_ip = request.client.host if request.client else "unknown"

    body = {}
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        body = await request.json()
    else:
        form = await request.form()
        body = dict(form)

    identifier = (body.get("email_or_username") or body.get("username") or body.get("email") or "").strip().lower()

    # Rate limiting on IP and IP:identifier
    keys_to_check = [client_ip]
    if identifier:
        keys_to_check.append(f"{client_ip}:{identifier}")

    for k in keys_to_check:
        if not check_login_rate_limit(k):
            return JSONResponse(
                {"status": "error", "message": "Too many failed login attempts. Please wait 60 seconds."},
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            )

    password = body.get("password") or ""
    remember = bool(body.get("remember_me", True))

    ok, user, msg = auth_service.authenticate(identifier, password)
    if not ok or not user:
        for k in keys_to_check:
            record_failed_attempt(k)
        audit_service.log(
            action="AUTH_FAIL",
            resource_type="auth",
            request=request,
            status="FAILED",
            severity="WARNING",
            details={"identifier": identifier, "reason": msg},
        )
        return JSONResponse({"status": "error", "message": msg}, status_code=status.HTTP_401_UNAUTHORIZED)

    for k in keys_to_check:
        clear_failed_attempts(k)

    user_agent = request.headers.get("user-agent")
    token = auth_service.create_session_token(
        user_id=user.id,
        ip_address=client_ip,
        user_agent=user_agent,
    )
    csrf_token = secrets.token_hex(16)

    audit_service.log(
        action="AUTH_LOGIN",
        resource_type="auth",
        resource_id=str(user.id),
        request=request,
        status="SUCCESS",
        severity="INFO",
        actor_user_id=str(user.id),
        actor_name=user.full_name or user.username,
        details={"username": user.username, "role": user.role},
    )

    response = JSONResponse({
        "status": "success",
        "message": "Signed in successfully",
        "csrf_token": csrf_token,
        "user": {
            "id": user.id,
            "email": user.email,
            "username": user.username,
            "full_name": user.full_name,
            "role": user.role,
        },
    })

    is_production = os.environ.get("DOCTUNE_ENV", "development").lower() == "production"
    max_age = (30 * 24 * 3600) if remember else (24 * 3600)

    response.set_cookie(
        key="doctune_session",
        value=token,
        max_age=max_age,
        httponly=True,
        secure=is_production,
        samesite="lax",
        path="/",
    )
    response.set_cookie(
        key="doctune_csrf",
        value=csrf_token,
        max_age=max_age,
        httponly=False,  # Accessible by JavaScript for double-submit header
        secure=is_production,
        samesite="lax",
        path="/",
    )
    response.set_cookie(
        key="doctune_operator",
        value=user.full_name or user.username,
        max_age=max_age,
        path="/",
    )
    return response


@router.post("/api/v1/auth/register", response_class=JSONResponse)
async def register_api(request: Request):
    """Register a new user account."""
    body = {}
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        body = await request.json()
    else:
        form = await request.form()
        body = dict(form)

    email = body.get("email", "")
    username = body.get("username", "")
    password = body.get("password", "")
    full_name = body.get("full_name", "")

    user_count = auth_service.count_users()
    is_first = (user_count == 0)
    auth_cfg = config_service.get_auth_config()

    curr_user = getattr(request.state, "user", None)
    if not is_first and not auth_cfg.get("allow_registration", True) and not (curr_user and curr_user.role == "admin"):
        return JSONResponse({"status": "error", "message": "Public registration is disabled on this server."}, status_code=403)

    ok, user, msg = auth_service.create_user(
        email=email,
        username=username,
        password=password,
        full_name=full_name,
    )

    if not ok or not user:
        return JSONResponse({"status": "error", "message": msg}, status_code=400)

    client_ip = request.client.host if request.client else "unknown"
    token = auth_service.create_session_token(
        user_id=user.id,
        ip_address=client_ip,
        user_agent=request.headers.get("user-agent"),
    )
    csrf_token = secrets.token_hex(16)

    audit_service.log(
        action="AUTH_REGISTER",
        resource_type="auth",
        resource_id=str(user.id),
        request=request,
        status="SUCCESS",
        severity="INFO",
        actor_user_id=str(user.id),
        actor_name=user.full_name or user.username,
        details={"username": user.username, "role": user.role, "is_first_user": is_first},
    )

    response = JSONResponse({
        "status": "success",
        "message": "Account created successfully.",
        "csrf_token": csrf_token,
        "user": {
            "id": user.id,
            "email": user.email,
            "username": user.username,
            "full_name": user.full_name,
            "role": user.role,
        },
    })

    is_production = os.environ.get("DOCTUNE_ENV", "development").lower() == "production"
    response.set_cookie(
        key="doctune_session",
        value=token,
        max_age=30 * 24 * 3600,
        httponly=True,
        secure=is_production,
        samesite="lax",
        path="/",
    )
    response.set_cookie(
        key="doctune_csrf",
        value=csrf_token,
        max_age=30 * 24 * 3600,
        httponly=False,
        secure=is_production,
        samesite="lax",
        path="/",
    )
    response.set_cookie(
        key="doctune_operator",
        value=user.full_name or user.username,
        max_age=30 * 24 * 3600,
        path="/",
    )
    return response


@router.post("/api/v1/auth/forgot-password", response_class=JSONResponse)
async def forgot_password_api(request: Request):
    """Generate password reset token."""
    body = {}
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        body = await request.json()
    else:
        form = await request.form()
        body = dict(form)

    email = body.get("email", "")
    ok, token, msg = auth_service.generate_reset_token(email)

    audit_service.log(
        action="AUTH_PASSWORD_RESET_REQ",
        resource_type="auth",
        request=request,
        status="SUCCESS" if ok else "FAILED",
        severity="INFO",
        details={"email": email},
    )

    resp_data: Dict[str, Any] = {"status": "success", "message": msg}
    if ok and token and os.environ.get("DOCTUNE_ENV") != "production":
        resp_data["dev_reset_token"] = token
        resp_data["dev_reset_link"] = f"/reset-password?token={token}"

    return JSONResponse(resp_data)


@router.post("/api/v1/auth/reset-password", response_class=JSONResponse)
async def reset_password_api(request: Request):
    """Reset password and revoke all prior sessions."""
    body = {}
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        body = await request.json()
    else:
        form = await request.form()
        body = dict(form)

    token = body.get("token", "")
    new_password = body.get("new_password") or body.get("password") or ""

    ok, user, msg = auth_service.reset_password_with_token(token, new_password)
    if not ok or not user:
        return JSONResponse({"status": "error", "message": msg}, status_code=400)

    audit_service.log(
        action="PASSWORD_RESET",
        resource_type="auth",
        resource_id=str(user.id),
        request=request,
        status="SUCCESS",
        severity="INFO",
        actor_user_id=str(user.id),
        actor_name=user.full_name or user.username,
    )

    return JSONResponse({"status": "success", "message": msg})


@router.post("/api/v1/auth/logout", response_class=JSONResponse)
async def logout_api(request: Request):
    """Terminate the active session in database and clear authentication cookies."""
    token = request.cookies.get("doctune_session")
    user = getattr(request.state, "user", None)

    if token:
        auth_service.revoke_session(token)

    if user:
        audit_service.log(
            action="AUTH_LOGOUT",
            resource_type="auth",
            resource_id=str(user.id),
            request=request,
            status="SUCCESS",
            actor_user_id=str(user.id),
            actor_name=user.full_name or user.username,
        )

    response = JSONResponse({"status": "success", "message": "Signed out successfully."})
    response.delete_cookie(key="doctune_session", path="/")
    response.delete_cookie(key="doctune_csrf", path="/")
    response.delete_cookie(key="doctune_operator", path="/")
    return response


@router.get("/api/v1/auth/me", response_class=JSONResponse)
async def get_current_user_profile(request: Request):
    """Return profile details and authorized projects for the active session."""
    user = getattr(request.state, "user", None)
    if not user:
        return JSONResponse({"authenticated": False, "user": None, "projects": []}, status_code=200)

    projects = auth_service.get_user_projects(user.id)
    active_project_id = request.cookies.get("doctune_project") or (projects[0]["id"] if projects else "proj_default")

    return JSONResponse({
        "authenticated": True,
        "user": {
            "id": user.id,
            "email": user.email,
            "username": user.username,
            "full_name": user.full_name,
            "role": user.role,
        },
        "active_project_id": active_project_id,
        "projects": projects,
    })


@router.get("/api/v1/auth/users", response_class=JSONResponse)
async def list_users_api(request: Request, user: User = Depends(require_site_admin)):
    """Return list of accounts (administrator-accessible)."""
    users = auth_service.list_users(limit=100)
    return {"status": "success", "count": len(users), "users": users}


# ── Multi-Tenant Project Management Endpoints ─────────────────────────────────

@router.get("/api/v1/projects", response_class=JSONResponse)
async def list_projects_api(request: Request, user: User = Depends(require_user)):
    """List all projects available to the active user."""
    projects = auth_service.get_user_projects(user.id)
    return {"status": "success", "projects": projects}


@router.post("/api/v1/projects/switch", response_class=JSONResponse)
async def switch_project_api(request: Request, user: User = Depends(require_user)):
    """Switch the current active workspace cookie."""
    body = {}
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        body = await request.json()
    else:
        form = await request.form()
        body = dict(form)

    project_id = body.get("project_id", "")
    role = auth_service.get_user_project_role(user.id, project_id)
    if not role:
        raise HTTPException(status_code=403, detail="You do not have access to this workspace.")

    response = JSONResponse({"status": "success", "active_project_id": project_id, "role": role})
    response.set_cookie(key="doctune_project", value=project_id, max_age=30 * 24 * 3600, path="/")
    return response


@router.post("/api/v1/projects", response_class=JSONResponse)
async def create_project_api(
    request: Request,
    user: User = Depends(require_user),
    db: Session = Depends(get_db),
):
    """Create a new isolated project workspace."""
    body = {}
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        body = await request.json()
    else:
        form = await request.form()
        body = dict(form)

    name = (body.get("name") or "").strip()
    slug = (body.get("slug") or "").strip().lower()
    description = body.get("description", "")

    if not name or len(name) < 2:
        return JSONResponse({"status": "error", "message": "Project name is required."}, status_code=400)

    if not slug:
        slug = "".join(c if c.isalnum() else "-" for c in name.lower()).strip("-")[:32]

    active_db, should_close = resolve_session(db)
    if not active_db:
        return JSONResponse({"status": "error", "message": "Database unavailable."}, status_code=503)

    try:
        existing = active_db.query(Project).filter(Project.slug == slug).first()
        if existing:
            return JSONResponse({"status": "error", "message": f"Project slug '{slug}' already exists."}, status_code=400)

        proj_id = f"proj_{secrets.token_hex(8)}"
        from datetime import datetime, timezone
        new_proj = Project(
            id=proj_id,
            name=name,
            slug=slug,
            description=description,
            storage_quota_bytes=53687091200,
            created_by=user.id,
            created_at=datetime.now(timezone.utc),
        )
        active_db.add(new_proj)
        active_db.flush()

        member = ProjectMember(
            project_id=new_proj.id,
            user_id=user.id,
            role="admin",
            created_at=datetime.now(timezone.utc),
        )
        active_db.add(member)
        active_db.commit()
        active_db.refresh(new_proj)

        response = JSONResponse({
            "status": "success",
            "message": f"Workspace '{name}' created successfully.",
            "project": {"id": new_proj.id, "name": new_proj.name, "slug": new_proj.slug, "role": "admin"},
        })
        response.set_cookie(key="doctune_project", value=new_proj.id, max_age=30 * 24 * 3600, path="/")
        return response
    except Exception as ex:
        active_db.rollback()
        logger.error("Failed creating project: %s", ex)
        return JSONResponse({"status": "error", "message": str(ex)}, status_code=500)
    finally:
        if should_close:
            active_db.close()
