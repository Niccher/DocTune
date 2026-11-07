"""
web/routers/auth.py
-------------------
Authentication endpoints for user registration, login, logout,
password recovery, and user session state in DocTune Studio.
"""

from typing import Dict, Any, Optional
from fastapi import APIRouter, Request, Response, HTTPException, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from web.services.auth_service import auth_service
from web.services.audit_service import audit_service
from web.services.config_service import config_service
from web.templates_config import templates

router = APIRouter(tags=["authentication"])


# ── UI Route Handlers ─────────────────────────────────────────────────────────

@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, next: str = "/"):
    """Render the user login page."""
    user = getattr(request.state, "user", None)
    if user:
        return RedirectResponse(url=next or "/", status_code=status.HTTP_302_FOUND)

    user_count = auth_service.count_users()
    if user_count == 0:
        # No users exist yet, guide to initial admin registration
        return RedirectResponse(url="/register", status_code=status.HTTP_302_FOUND)

    return templates.TemplateResponse(
        request=request,
        name="auth/login.html",
        context={
            "page_title": "Sign In — DocTune Studio",
            "next_url": next or "/",
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
async def forgot_password_page(request: Request):
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
            "token": token or "",
            "app_name": config_service.get_app_name(),
        },
    )


# ── REST API Endpoints ────────────────────────────────────────────────────────

@router.post("/api/v1/auth/login", response_class=JSONResponse)
async def login_api(request: Request):
    """Authenticate credentials and establish signed session cookie."""
    body = {}
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        body = await request.json()
    else:
        form = await request.form()
        body = dict(form)

    identifier = body.get("email_or_username") or body.get("username") or body.get("email") or ""
    password = body.get("password") or ""
    remember = bool(body.get("remember_me", True))

    ok, user, msg = auth_service.authenticate(identifier, password)
    if not ok or not user:
        audit_service.log(
            action="AUTH_FAIL",
            resource_type="auth",
            request=request,
            status="FAILED",
            severity="WARNING",
            details={"identifier": identifier, "reason": msg},
        )
        return JSONResponse({"status": "error", "message": msg}, status_code=status.HTTP_401_UNAUTHORIZED)

    token = auth_service.create_session_token(user.id)

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
        "user": {
            "id": user.id,
            "email": user.email,
            "username": user.username,
            "full_name": user.full_name,
            "role": user.role,
        },
    })

    max_age = (30 * 24 * 3600) if remember else (24 * 3600)
    response.set_cookie(
        key="doctune_session",
        value=token,
        max_age=max_age,
        httponly=True,
        samesite="lax",
        path="/",
    )
    # Set display operator tag for UI alignment
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

    token = auth_service.create_session_token(user.id)

    audit_service.log(
        action="AUTH_REGISTER",
        resource_type="auth",
        resource_id=str(user.id),
        request=request,
        status="SUCCESS",
        severity="INFO",
        actor_user_id=str(user.id),
        actor_name=user.full_name or user.username,
        details={"username": user.username, "role": user.role, "is_admin": user.role == "admin"},
    )

    response = JSONResponse({
        "status": "success",
        "message": "Account created successfully",
        "user": {
            "id": user.id,
            "email": user.email,
            "username": user.username,
            "full_name": user.full_name,
            "role": user.role,
        },
    })
    response.set_cookie(
        key="doctune_session",
        value=token,
        max_age=30 * 24 * 3600,
        httponly=True,
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
    """Generate a password recovery token."""
    body = {}
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        body = await request.json()
    else:
        form = await request.form()
        body = dict(form)

    email = body.get("email", "")
    ok, token, msg = auth_service.generate_reset_token(email)

    base = str(request.base_url).rstrip("/")
    reset_url = f"{base}/reset-password?token={token}" if token else None

    audit_service.log(
        action="AUTH_RESET_REQ",
        resource_type="auth",
        request=request,
        status="SUCCESS" if ok else "FAILED",
        details={"email": email, "token_generated": bool(token)},
    )

    return JSONResponse({
        "status": "success",
        "message": msg,
        "token": token,
        "reset_url": reset_url,
    })


@router.post("/api/v1/auth/reset-password", response_class=JSONResponse)
async def reset_password_api(request: Request):
    """Reset password using a valid recovery token."""
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
    """Terminate the active session and revoke the authentication cookie."""
    user = getattr(request.state, "user", None)
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
    return response


@router.get("/api/v1/auth/me", response_class=JSONResponse)
async def get_current_user_profile(request: Request):
    """Return profile details for the active session."""
    user = getattr(request.state, "user", None)
    if not user:
        return JSONResponse({"authenticated": False, "user": None}, status_code=200)

    return JSONResponse({
        "authenticated": True,
        "user": {
            "id": user.id,
            "email": user.email,
            "username": user.username,
            "full_name": user.full_name,
            "role": user.role,
        },
    })


@router.get("/api/v1/auth/users", response_class=JSONResponse)
async def list_users_api(request: Request):
    """Return list of accounts (admin-accessible)."""
    user = getattr(request.state, "user", None)
    auth_cfg = config_service.get_auth_config()

    # If auth is enabled, only admins can view the full user list
    if auth_cfg.get("enabled", False) and not (user and user.role == "admin"):
        raise HTTPException(status_code=403, detail="Administrator permissions required.")

    users = auth_service.list_users(limit=100)
    return {"status": "success", "count": len(users), "users": users}
