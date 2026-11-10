"""
web/services/auth_service.py
----------------------------
Zero-dependency authentication, password security, session management,
and multi-tenant project authorization service.
Provides PBKDF2-HMAC-SHA256 password hashing, server-side cryptographic
session token management with revocation, provider credential encryption,
and role-based project access control.
"""

import os
import hmac
import time
import base64
import logging
import hashlib
import secrets
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone, timedelta

from sqlalchemy.orm import Session
from web.db.session import SessionLocal, HAS_SQLALCHEMY
from web.db.models import User, UserSession, Project, ProjectMember, ApiKey, ProviderCredential
from web.services.config_service import config_service

logger = logging.getLogger(__name__)

DEFAULT_SECRET = "doctune-default-secure-secret-key-change-in-production"

try:
    from cryptography.fernet import Fernet
    HAS_FERNET = True
except ImportError:
    Fernet = None
    HAS_FERNET = False


def hash_password(password: str) -> str:
    """NIST-compliant zero-dependency password hash using PBKDF2-HMAC-SHA256."""
    salt = secrets.token_hex(16)
    key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100000)
    return f"{salt}${key.hex()}"


def verify_password(password: str, hashed: str) -> bool:
    """Constant-time verification of raw password against salt$hash string."""
    try:
        salt, key_hex = hashed.split("$", 1)
        key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100000)
        return hmac.compare_digest(key.hex(), key_hex)
    except Exception:
        return False


class AuthService:
    """Enterprise authentication, multi-tenant RBAC, and session lifecycle service."""

    hash_password = staticmethod(hash_password)
    verify_password = staticmethod(verify_password)

    def __init__(self):
        self._fallback_secret = secrets.token_hex(32)
        self.validate_startup_security()

    def validate_startup_security(self):
        """Refuse to start or emit critical warning if SECRET_KEY is insecure in production."""
        env = os.environ.get("DOCTUNE_ENV", "development").lower()
        key = os.environ.get("SECRET_KEY") or os.environ.get("AUTH_SECRET_KEY")
        if env == "production":
            if not key or key.strip() == "" or key.strip() == DEFAULT_SECRET or "change-this" in key.lower():
                raise RuntimeError(
                    "CRITICAL SECURITY CONFIGURATION ERROR: DOCTUNE_ENV=production requires a secure, "
                    "random SECRET_KEY environment variable. Refusing to boot with default/missing credentials."
                )
        elif not key or key.strip() == DEFAULT_SECRET:
            logger.warning("Running with default/temporary SECRET_KEY. Set SECRET_KEY in your .env for production.")

    def get_secret_key(self) -> str:
        """Retrieve persistent signing secret key from environment or settings."""
        env_secret = os.environ.get("SECRET_KEY") or os.environ.get("AUTH_SECRET_KEY")
        if env_secret and env_secret.strip() and "change-this" not in env_secret.lower():
            return env_secret.strip()
        cfg = config_service.get_auth_config()
        key = cfg.get("secret_key")
        if key and key.strip() and key != DEFAULT_SECRET:
            return key.strip()
        return self._fallback_secret

    # ── Server-Side Session Management ────────────────────────────────────────

    def create_session_token(
        self,
        user_id: int,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        db: Optional[Session] = None,
    ) -> str:
        """
        Generate a cryptographically secure 256-bit token, store its SHA-256 hash
        in the sessions table with metadata, and return the client cookie token.
        """
        raw_secret = secrets.token_urlsafe(32)
        token_hash = hashlib.sha256(raw_secret.encode("utf-8")).hexdigest()
        session_id = f"sess_{secrets.token_hex(12)}"

        cfg = config_service.get_auth_config()
        expire_hours = int(cfg.get("session_expire_hours", 168))  # default 7 days
        expires_at = datetime.now(timezone.utc) + timedelta(hours=expire_hours)

        close_session = False
        if db is None and HAS_SQLALCHEMY and SessionLocal:
            db = SessionLocal()
            close_session = True

        if db:
            try:
                session_record = UserSession(
                    id=session_id,
                    token_hash=token_hash,
                    user_id=user_id,
                    ip_address=ip_address,
                    user_agent=(user_agent or "")[:256],
                    expires_at=expires_at,
                    revoked_at=None,
                    created_at=datetime.now(timezone.utc),
                )
                db.add(session_record)
                db.commit()
            except Exception as ex:
                logger.error("Failed persisting session record: %s", ex)
                if db:
                    db.rollback()
            finally:
                if close_session and db:
                    db.close()

        # Format: session_id.raw_secret
        return f"{session_id}.{raw_secret}"

    def verify_session_token(self, token: Optional[str], db: Optional[Session] = None) -> Optional[int]:
        """
        Verify server-side session token against hashed database record.
        Ensures the token exists, is unexpired, and has not been revoked.
        """
        if not token or not isinstance(token, str):
            return None

        parts = token.split(".", 1)
        if len(parts) != 2:
            # Fallback to legacy HMAC token check during rolling updates
            return self._verify_legacy_hmac_token(token)

        session_id, raw_secret = parts
        token_hash = hashlib.sha256(raw_secret.encode("utf-8")).hexdigest()

        close_session = False
        if db is None and HAS_SQLALCHEMY and SessionLocal:
            db = SessionLocal()
            close_session = True

        if not db:
            return None

        try:
            record = db.query(UserSession).filter(
                UserSession.id == session_id,
                UserSession.token_hash == token_hash,
            ).first()

            if not record:
                return None

            if record.revoked_at is not None:
                return None

            now = datetime.now(timezone.utc)
            exp = record.expires_at
            if exp.tzinfo is None:
                exp = exp.replace(tzinfo=timezone.utc)
            if now > exp:
                return None

            return record.user_id
        except Exception as e:
            logger.debug("Session verification error: %s", e)
            return None
        finally:
            if close_session and db:
                db.close()

    def _verify_legacy_hmac_token(self, token: str) -> Optional[int]:
        """Backward compatibility for legacy tokens during rollout."""
        parts = token.split(".")
        if len(parts) != 3:
            return None
        user_id_str, ts_str, signature = parts
        secret = self.get_secret_key()
        payload = f"{user_id_str}.{ts_str}"
        expected_sig = hmac.new(secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()[:32]
        if not hmac.compare_digest(signature, expected_sig):
            return None
        try:
            created_at = int(ts_str)
            user_id = int(user_id_str)
            cfg = config_service.get_auth_config()
            expire_hours = int(cfg.get("session_expire_hours", 168))
            if (time.time() - created_at) > (expire_hours * 3600):
                return None
            return user_id
        except (ValueError, TypeError):
            return None

    def revoke_session(self, token: Optional[str], db: Optional[Session] = None) -> bool:
        """Revoke a specific active session token."""
        if not token or "." not in token:
            return False

        parts = token.split(".", 1)
        if len(parts) != 2:
            return False

        session_id, raw_secret = parts
        token_hash = hashlib.sha256(raw_secret.encode("utf-8")).hexdigest()

        close_session = False
        if db is None and HAS_SQLALCHEMY and SessionLocal:
            db = SessionLocal()
            close_session = True

        if not db:
            return False

        try:
            record = db.query(UserSession).filter(
                UserSession.id == session_id,
                UserSession.token_hash == token_hash,
            ).first()
            if record:
                record.revoked_at = datetime.now(timezone.utc)
                db.commit()
                return True
            return False
        except Exception as ex:
            logger.error("Failed revoking session: %s", ex)
            if db:
                db.rollback()
            return False
        finally:
            if close_session and db:
                db.close()

    validate_session_token = verify_session_token
    revoke_session_token = revoke_session

    def generate_csrf_token(self, token_context: Optional[str] = None) -> str:
        """Generate a cryptographically random CSRF token (optionally keyed to session)."""
        if token_context:
            secret = self.get_secret_key()
            return hmac.new(secret.encode("utf-8"), token_context.encode("utf-8"), hashlib.sha256).hexdigest()[:32]
        return secrets.token_hex(16)

    def validate_csrf_token(self, submitted_csrf: Optional[str], expected_csrf: Optional[str]) -> bool:
        """Validate double-submit or HMAC-derived CSRF token against expected value."""
        if not submitted_csrf or not expected_csrf:
            return False
        # If expected is a session token and submitted is HMAC, verify
        if "." in expected_csrf and submitted_csrf == self.generate_csrf_token(expected_csrf):
            return True
        return hmac.compare_digest(submitted_csrf, expected_csrf)

    def create_api_key(
        self,
        user_id: int,
        project_id: str = "proj_default",
        name: str = "API Key",
        db: Optional[Session] = None,
    ) -> Tuple[str, Any]:
        """Create a live API key and store its SHA-256 hash in database."""
        raw_key = f"dt_live_{secrets.token_urlsafe(32)}"
        key_hash = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
        prefix = raw_key[:12]
        key_id = f"key_{secrets.token_hex(8)}"

        close_session = False
        if db is None and HAS_SQLALCHEMY and SessionLocal:
            db = SessionLocal()
            close_session = True

        if db:
            try:
                key_record = ApiKey(
                    id=key_id,
                    key_hash=key_hash,
                    prefix=prefix,
                    user_id=user_id,
                    project_id=project_id,
                    name=name,
                    revoked_at=None,
                )
                db.add(key_record)
                db.commit()
                return raw_key, key_record
            finally:
                if close_session and db:
                    db.close()
        return raw_key, None

    def validate_api_key(
        self,
        key: str,
        db: Optional[Session] = None,
    ) -> Tuple[Optional[Any], Optional[str]]:
        """Validate API key token and return (User, project_id)."""
        if not key:
            return None, None
        clean = key.replace("Bearer ", "").strip()
        key_hash = hashlib.sha256(clean.encode("utf-8")).hexdigest()

        close_session = False
        if db is None and HAS_SQLALCHEMY and SessionLocal:
            db = SessionLocal()
            close_session = True

        if db:
            try:
                rec = db.query(ApiKey).filter(ApiKey.key_hash == key_hash, ApiKey.revoked_at.is_(None)).first()
                if rec:
                    user = db.query(User).filter(User.id == rec.user_id).first()
                    return user, rec.project_id
            finally:
                if close_session and db:
                    db.close()
        return None, None

    def revoke_all_user_sessions(self, user_id: int, db: Optional[Session] = None) -> int:
        """Revoke all active sessions for a user (called on password change/reset)."""
        if not HAS_SQLALCHEMY or not SessionLocal:
            return 0

        close_session = False
        if db is None:
            db = SessionLocal()
            close_session = True

        try:
            now = datetime.now(timezone.utc)
            count = db.query(UserSession).filter(
                UserSession.user_id == user_id,
                UserSession.revoked_at.is_(None),
            ).update({"revoked_at": now})
            db.commit()
            return count
        except Exception as ex:
            logger.error("Failed revoking user sessions: %s", ex)
            if db:
                db.rollback()
            return 0
        finally:
            if close_session and db:
                db.close()

    # ── Provider Credential Encryption (Fernet / Authenticated Stream) ────────

    def _get_encryption_key(self) -> bytes:
        """Derive 32-byte key for credential encryption."""
        raw = os.environ.get("ENCRYPTION_KEY") or self.get_secret_key()
        return hashlib.sha256(raw.encode("utf-8")).digest()

    def encrypt_credential(self, plaintext: str) -> str:
        """Encrypt third-party API key with Fernet (or HMAC-authenticated keystream)."""
        if not plaintext:
            return ""
        key_32 = self._get_encryption_key()
        if HAS_FERNET and Fernet:
            f_key = base64.urlsafe_b64encode(key_32)
            f = Fernet(f_key)
            return f.encrypt(plaintext.encode("utf-8")).decode("utf-8")

        # Zero-dependency authenticated encryption fallback
        nonce = secrets.token_bytes(16)
        stream_key = hmac.new(key_32, b"stream:" + nonce, hashlib.sha256).digest()
        cipher_bytes = bytes(b ^ stream_key[i % len(stream_key)] for i, b in enumerate(plaintext.encode("utf-8")))
        tag = hmac.new(key_32, b"tag:" + nonce + cipher_bytes, hashlib.sha256).digest()[:16]
        payload = nonce + tag + cipher_bytes
        return "b64:" + base64.urlsafe_b64encode(payload).decode("utf-8")

    def decrypt_credential(self, ciphertext: str) -> str:
        """Decrypt third-party API key."""
        if not ciphertext:
            return ""
        key_32 = self._get_encryption_key()
        if HAS_FERNET and Fernet and not ciphertext.startswith("b64:"):
            try:
                f_key = base64.urlsafe_b64encode(key_32)
                f = Fernet(f_key)
                return f.decrypt(ciphertext.encode("utf-8")).decode("utf-8")
            except Exception:
                pass

        # Zero-dependency decrypt fallback
        try:
            raw_b64 = ciphertext[4:] if ciphertext.startswith("b64:") else ciphertext
            payload = base64.urlsafe_b64decode(raw_b64.encode("utf-8"))
            nonce = payload[:16]
            tag = payload[16:32]
            cipher_bytes = payload[32:]
            expected_tag = hmac.new(key_32, b"tag:" + nonce + cipher_bytes, hashlib.sha256).digest()[:16]
            if not hmac.compare_digest(tag, expected_tag):
                return ""
            stream_key = hmac.new(key_32, b"stream:" + nonce, hashlib.sha256).digest()
            plain_bytes = bytes(b ^ stream_key[i % len(stream_key)] for i, b in enumerate(cipher_bytes))
            return plain_bytes.decode("utf-8")
        except Exception:
            return ""

    # ── Multi-Tenant Project Authorization ────────────────────────────────────

    def get_or_create_default_project(self, db: Optional[Session] = None) -> Optional[Project]:
        """Ensure the baseline 'Default Workspace' exists."""
        if not HAS_SQLALCHEMY or not SessionLocal:
            return None

        close_session = False
        if db is None:
            db = SessionLocal()
            close_session = True

        try:
            proj = db.query(Project).filter(Project.id == "proj_default").first()
            if not proj:
                proj = Project(
                    id="proj_default",
                    name="Default Workspace",
                    slug="default",
                    description="Default fine-tuning workspace",
                    storage_quota_bytes=53687091200,
                    created_at=datetime.now(timezone.utc),
                )
                db.add(proj)
                db.commit()
                db.refresh(proj)
            return proj
        except Exception as ex:
            logger.error("Failed ensuring default project: %s", ex)
            if db:
                db.rollback()
            return None
        finally:
            if close_session and db:
                db.close()

    def get_user_projects(self, user_id: int, db: Optional[Session] = None) -> List[Dict[str, Any]]:
        """Return list of projects the user is authorized to access."""
        if not HAS_SQLALCHEMY or not SessionLocal:
            return []

        close_session = False
        if db is None:
            db = SessionLocal()
            close_session = True

        try:
            user = db.query(User).filter(User.id == user_id).first()
            if not user:
                return []

            if user.role == "admin":
                # System admins can access all projects
                projects = db.query(Project).all()
                return [
                    {
                        "id": p.id,
                        "name": p.name,
                        "slug": p.slug,
                        "role": "admin",
                        "storage_quota_bytes": p.storage_quota_bytes,
                    }
                    for p in projects
                ]

            memberships = db.query(ProjectMember).filter(ProjectMember.user_id == user_id).all()
            result = []
            for m in memberships:
                p = m.project
                if p:
                    result.append({
                        "id": p.id,
                        "name": p.name,
                        "slug": p.slug,
                        "role": m.role,
                        "storage_quota_bytes": p.storage_quota_bytes,
                    })

            # Always ensure membership in Default Workspace
            if not any(r["id"] == "proj_default" for r in result):
                def_proj = self.get_or_create_default_project(db)
                if def_proj:
                    result.insert(0, {
                        "id": def_proj.id,
                        "name": def_proj.name,
                        "slug": def_proj.slug,
                        "role": user.role,
                        "storage_quota_bytes": def_proj.storage_quota_bytes,
                    })
            return result
        except Exception as ex:
            logger.error("Error retrieving user projects: %s", ex)
            return []
        finally:
            if close_session and db:
                db.close()

    def get_user_project_role(self, user_id: int, project_id: str, db: Optional[Session] = None) -> Optional[str]:
        """
        Return the user's role in a project ('admin', 'operator', 'viewer'),
        or None if the user has no access.
        """
        if not HAS_SQLALCHEMY or not SessionLocal:
            return None

        close_session = False
        if db is None:
            db = SessionLocal()
            close_session = True

        try:
            user = db.query(User).filter(User.id == user_id).first()
            if not user or not user.is_active:
                return None

            if user.role == "admin":
                return "admin"

            member = db.query(ProjectMember).filter(
                ProjectMember.user_id == user_id,
                ProjectMember.project_id == project_id,
            ).first()

            if member:
                return member.role

            # Fallback for Default Workspace
            if project_id == "proj_default":
                return user.role or "operator"

            return None
        except Exception as ex:
            logger.error("Error checking project role: %s", ex)
            return None
        finally:
            if close_session and db:
                db.close()

    # ── User Lifecycle Management ─────────────────────────────────────────────

    def count_users(self, db: Optional[Session] = None) -> int:
        """Return total registered user accounts."""
        if not HAS_SQLALCHEMY or not SessionLocal:
            return 0
        close_session = False
        if db is None:
            db = SessionLocal()
            close_session = True
        try:
            return db.query(User).count()
        except Exception:
            return 0
        finally:
            if close_session and db:
                db.close()

    def create_user(
        self,
        email: str,
        username: str,
        password: str,
        full_name: Optional[str] = None,
        role: Optional[str] = None,
        db: Optional[Session] = None,
    ) -> Tuple[bool, Optional[User], str]:
        """Register a new user account. First registered user is automatically granted 'admin' role."""
        if not HAS_SQLALCHEMY or not SessionLocal:
            return False, None, "Database service unavailable."

        email_clean = (email or "").strip().lower()
        username_clean = (username or "").strip().lower()

        if not email_clean or "@" not in email_clean:
            return False, None, "A valid email address is required."
        if not username_clean or len(username_clean) < 3:
            return False, None, "Username must be at least 3 characters long."
        if not password or len(password) < 8:
            return False, None, "Password must be at least 8 characters long."

        close_session = False
        if db is None:
            db = SessionLocal()
            close_session = True

        try:
            existing = db.query(User).filter((User.email == email_clean) | (User.username == username_clean)).first()
            if existing:
                if existing.email == email_clean:
                    return False, None, f"An account with email '{email_clean}' already exists."
                return False, None, f"Username '{username_clean}' is already taken."

            total_users = db.query(User).count()
            assigned_role = role or ("admin" if total_users == 0 else "operator")

            hashed = hash_password(password)
            user = User(
                email=email_clean,
                username=username_clean,
                full_name=(full_name or "").strip() or username_clean.capitalize(),
                hashed_password=hashed,
                role=assigned_role,
                is_active=True,
                created_at=datetime.now(timezone.utc),
            )
            db.add(user)
            db.commit()
            db.refresh(user)

            # Auto-assign membership in default project
            def_proj = self.get_or_create_default_project(db)
            if def_proj:
                member = ProjectMember(
                    project_id=def_proj.id,
                    user_id=user.id,
                    role=assigned_role,
                    created_at=datetime.now(timezone.utc),
                )
                db.add(member)
                db.commit()

            logger.info("Created user account '%s' (role: %s)", user.username, user.role)
            return True, user, "Account created successfully."
        except Exception as e:
            logger.error("Failed to create user account: %s", e)
            if db:
                db.rollback()
            return False, None, f"Failed creating account: {str(e)}"
        finally:
            if close_session and db:
                db.close()

    def authenticate(
        self,
        email_or_username: str,
        password: str,
        db: Optional[Session] = None,
    ) -> Tuple[bool, Optional[User], str]:
        """Validate user credentials and return User record if authenticated."""
        if not HAS_SQLALCHEMY or not SessionLocal:
            return False, None, "Database service unavailable."

        identifier = (email_or_username or "").strip().lower()
        if not identifier or not password:
            return False, None, "Email/Username and password are required."

        close_session = False
        if db is None:
            db = SessionLocal()
            close_session = True

        try:
            user = db.query(User).filter(
                (User.email == identifier) | (User.username == identifier)
            ).first()

            if not user:
                return False, None, "Invalid credentials."

            if not user.is_active:
                return False, None, "Account is disabled. Contact your administrator."

            if not verify_password(password, user.hashed_password):
                return False, None, "Invalid credentials."

            user.last_login_at = datetime.now(timezone.utc)
            db.commit()
            db.refresh(user)
            logger.info("User '%s' authenticated successfully.", user.username)
            return True, user, "Authentication successful."
        except Exception as e:
            logger.error("Authentication error: %s", e)
            if db:
                db.rollback()
            return False, None, "Authentication service error."
        finally:
            if close_session and db:
                db.close()

    def get_user_by_id(self, user_id: int, db: Optional[Session] = None) -> Optional[User]:
        """Retrieve user by primary key ID."""
        if not HAS_SQLALCHEMY or not SessionLocal or not user_id:
            return None
        close_session = False
        if db is None:
            db = SessionLocal()
            close_session = True
        try:
            return db.query(User).filter(User.id == user_id).first()
        except Exception:
            return None
        finally:
            if close_session and db:
                db.close()

    def get_user_by_email(self, email: str, db: Optional[Session] = None) -> Optional[User]:
        """Retrieve user by email address."""
        if not HAS_SQLALCHEMY or not SessionLocal or not email:
            return None
        close_session = False
        if db is None:
            db = SessionLocal()
            close_session = True
        try:
            return db.query(User).filter(User.email == email.strip().lower()).first()
        except Exception:
            return None
        finally:
            if close_session and db:
                db.close()

    def generate_reset_token(self, email: str, db: Optional[Session] = None) -> Tuple[bool, Optional[str], str]:
        """Generate a one-hour password reset token for the given account."""
        if not HAS_SQLALCHEMY or not SessionLocal:
            return False, None, "Database service unavailable."

        close_session = False
        if db is None:
            db = SessionLocal()
            close_session = True

        try:
            user = db.query(User).filter(User.email == email.strip().lower()).first()
            if not user:
                return False, None, f"If an account exists for '{email}', a reset link has been issued."

            token = secrets.token_urlsafe(32)
            user.reset_token = token
            user.reset_token_expires_at = datetime.now(timezone.utc) + timedelta(hours=1)
            db.commit()
            logger.info("Password reset token generated for user '%s'", user.username)
            return True, token, "Password reset token successfully generated."
        except Exception as e:
            logger.error("Failed generating reset token: %s", e)
            if db:
                db.rollback()
            return False, None, "Failed generating reset token."
        finally:
            if close_session and db:
                db.close()

    def reset_password_with_token(
        self,
        token: str,
        new_password: str,
        db: Optional[Session] = None,
    ) -> Tuple[bool, Optional[User], str]:
        """Update password using a valid, unexpired reset token and revoke all prior sessions."""
        if not HAS_SQLALCHEMY or not SessionLocal:
            return False, None, "Database service unavailable."

        if not token or not token.strip():
            return False, None, "Reset token is missing."
        if not new_password or len(new_password) < 8:
            return False, None, "New password must be at least 8 characters long."

        close_session = False
        if db is None:
            db = SessionLocal()
            close_session = True

        try:
            user = db.query(User).filter(User.reset_token == token.strip()).first()
            if not user:
                return False, None, "Invalid or expired reset token."

            now = datetime.now(timezone.utc)
            if user.reset_token_expires_at:
                exp = user.reset_token_expires_at
                if exp.tzinfo is None:
                    exp = exp.replace(tzinfo=timezone.utc)
                if now > exp:
                    return False, None, "Reset token has expired. Please request a new link."

            user.hashed_password = hash_password(new_password)
            user.reset_token = None
            user.reset_token_expires_at = None
            db.commit()
            db.refresh(user)

            # Revoke all existing sessions for this user on password reset
            self.revoke_all_user_sessions(user.id, db)

            logger.info("Password successfully updated and sessions revoked for user '%s'", user.username)
            return True, user, "Password has been successfully reset. You may now sign in."
        except Exception as e:
            logger.error("Failed resetting password: %s", e)
            if db:
                db.rollback()
            return False, None, "Failed resetting password."
        finally:
            if close_session and db:
                db.close()

    def list_users(self, limit: int = 100, db: Optional[Session] = None) -> List[Dict[str, Any]]:
        """List user accounts without sensitive password hashes."""
        if not HAS_SQLALCHEMY or not SessionLocal:
            return []
        close_session = False
        if db is None:
            db = SessionLocal()
            close_session = True
        try:
            users = db.query(User).order_by(User.created_at.asc()).limit(limit).all()
            return [
                {
                    "id": u.id,
                    "email": u.email,
                    "username": u.username,
                    "full_name": u.full_name,
                    "role": u.role,
                    "is_active": u.is_active,
                    "created_at": u.created_at.isoformat() if u.created_at else None,
                    "last_login_at": u.last_login_at.isoformat() if u.last_login_at else None,
                }
                for u in users
            ]
        except Exception:
            return []
        finally:
            if close_session and db:
                db.close()


auth_service = AuthService()
