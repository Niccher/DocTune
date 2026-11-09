"""
web/services/auth_service.py
----------------------------
Zero-dependency authentication, password security, and session management service.
Provides PBKDF2-HMAC-SHA256 password hashing, tamper-proof session signatures,
user lifecycle management, and password recovery token handling.
"""

import hmac
import time
import logging
import hashlib
import secrets
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone, timedelta

from sqlalchemy.orm import Session
from web.db.session import SessionLocal, HAS_SQLALCHEMY
from web.db.models import User
from web.services.config_service import config_service

logger = logging.getLogger(__name__)


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
    """Enterprise authentication and user management service."""

    hash_password = staticmethod(hash_password)
    verify_password = staticmethod(verify_password)

    def __init__(self):
        self._fallback_secret = secrets.token_hex(32)

    def get_secret_key(self) -> str:
        """Retrieve persistent signing secret key from environment or settings."""
        import os
        env_secret = os.environ.get("SECRET_KEY") or os.environ.get("AUTH_SECRET_KEY")
        if env_secret and env_secret.strip() and "change-this" not in env_secret.lower():
            return env_secret.strip()
        cfg = config_service.get_auth_config()
        key = cfg.get("secret_key")
        if key and key.strip() and key != "doctune-default-secure-secret-key-change-in-production":
            return key.strip()
        return self._fallback_secret

    def create_session_token(self, user_id: int) -> str:
        """Generate a signed, tamper-proof session token."""
        now = int(time.time())
        secret = self.get_secret_key()
        payload = f"{user_id}.{now}"
        signature = hmac.new(secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()[:32]
        return f"{payload}.{signature}"

    def verify_session_token(self, token: Optional[str]) -> Optional[int]:
        """Verify session token signature and expiration, returning user_id if valid."""
        if not token or not isinstance(token, str):
            return None

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
            # Check for existing email or username
            existing = db.query(User).filter((User.email == email_clean) | (User.username == username_clean)).first()
            if existing:
                if existing.email == email_clean:
                    return False, None, f"An account with email '{email_clean}' already exists."
                return False, None, f"Username '{username_clean}' is already taken."

            # Automatically promote first user to admin
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
                # Security: generic success message to prevent user enumeration
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
        """Update password using a valid, unexpired reset token."""
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
            logger.info("Password successfully updated for user '%s'", user.username)
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
