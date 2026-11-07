"""
web/services/audit_service.py
-----------------------------
Audit logging service with 3-tier actor resolution (IP, Operator Tag, and User ID).
Provides non-blocking, append-only operational event recording across DocTune Studio.
"""

import json
import logging
import uuid
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timezone

from fastapi import Request
from sqlalchemy.orm import Session

from web.db.session import SessionLocal, HAS_SQLALCHEMY
from web.db.models import AuditLog

logger = logging.getLogger(__name__)


class AuditService:
    """Enterprise audit logger providing non-blocking compliance tracking."""

    @staticmethod
    def resolve_actor(request: Optional[Request] = None) -> Tuple[str, str, Optional[str]]:
        """
        Extract caller identity using the 3-tier resolution strategy:
        1. Network IP (X-Forwarded-For or client host)
        2. Operator Tag (X-Operator-Name header, cookie, or anonymous)
        3. Future User ID (request.state or X-User-ID)
        """
        actor_ip = "127.0.0.1"
        actor_name = "System/Anonymous"
        actor_user_id = None

        if request:
            # 1. IP Resolution
            forwarded = request.headers.get("x-forwarded-for")
            if forwarded:
                actor_ip = forwarded.split(",")[0].strip()
            elif request.client and request.client.host:
                actor_ip = request.client.host

            # 2. Operator Tag Resolution (header takes priority, then cookie)
            op_header = request.headers.get("x-operator-name")
            op_cookie = request.cookies.get("doctune_operator")
            if op_header and op_header.strip():
                actor_name = op_header.strip()
            elif op_cookie and op_cookie.strip():
                actor_name = op_cookie.strip()

            # 3. Future User Authentication Hook
            if hasattr(request.state, "user_id"):
                actor_user_id = str(request.state.user_id)
            elif "x-user-id" in request.headers:
                actor_user_id = request.headers["x-user-id"].strip()

        return actor_ip, actor_name, actor_user_id

    @classmethod
    def extract_request_context(cls, request: Optional[Request] = None) -> Dict[str, Any]:
        """
        Extract rich telemetry context from incoming HTTP request.
        Includes actor identity, user agent, request ID, HTTP method, and path.
        """
        actor_ip, actor_name, actor_user_id = cls.resolve_actor(request)
        user_agent = None
        request_id = None
        request_method = None
        endpoint = None

        if request:
            # User agent (clipped to 256 chars)
            ua = request.headers.get("user-agent")
            if ua:
                user_agent = ua.strip()[:256]

            # Request ID correlation (header > state > generated uuid)
            req_id_header = request.headers.get("x-request-id")
            req_id_state = getattr(request.state, "request_id", None)
            if req_id_header and req_id_header.strip():
                request_id = req_id_header.strip()[:64]
            elif req_id_state:
                request_id = str(req_id_state)[:64]
            else:
                request_id = f"req-{uuid.uuid4().hex[:12]}"

            # HTTP Method and Endpoint Path
            if hasattr(request, "method") and request.method:
                request_method = request.method[:16].upper()
            if hasattr(request, "url") and hasattr(request.url, "path"):
                endpoint = request.url.path[:256]

        return {
            "actor_ip": actor_ip,
            "actor_name": actor_name,
            "actor_user_id": actor_user_id,
            "user_agent": user_agent,
            "request_id": request_id,
            "request_method": request_method,
            "endpoint": endpoint,
        }

    def log(
        self,
        action: str,
        resource_type: str,
        resource_id: Optional[str] = None,
        request: Optional[Request] = None,
        actor_ip: Optional[str] = None,
        actor_name: Optional[str] = None,
        actor_user_id: Optional[str] = None,
        status: str = "SUCCESS",
        severity: Optional[str] = None,
        request_id: Optional[str] = None,
        request_method: Optional[str] = None,
        endpoint: Optional[str] = None,
        user_agent: Optional[str] = None,
        duration_ms: Optional[float] = None,
        details: Optional[Union[Dict[str, Any], str]] = None,
        db: Optional[Session] = None,
    ) -> Optional[int]:
        """
        Record an operational event in the audit trail.
        Never raises exceptions to callers; non-blocking and fault-tolerant.
        """
        if not HAS_SQLALCHEMY or not SessionLocal:
            return None

        # Resolve actor & telemetry context if request is provided
        if request:
            ctx = self.extract_request_context(request)
            actor_ip = actor_ip or ctx["actor_ip"]
            actor_name = actor_name or ctx["actor_name"]
            actor_user_id = actor_user_id or ctx["actor_user_id"]
            user_agent = user_agent or ctx["user_agent"]
            request_id = request_id or ctx["request_id"]
            request_method = request_method or ctx["request_method"]
            endpoint = endpoint or ctx["endpoint"]

        actor_ip = (actor_ip or "127.0.0.1")[:64]
        actor_name = (actor_name or "System/Anonymous")[:128]
        if not request_id:
            request_id = f"op-{uuid.uuid4().hex[:12]}"

        # Intelligent severity classification if not explicitly provided
        if not severity:
            stat_up = (status or "SUCCESS").upper()
            act_up = (action or "").upper()
            if stat_up in ("FAILED", "ERROR"):
                severity = "WARNING"
            elif act_up in ("DOC_DELETE", "CONFIG_RESET", "MODEL_DELETE"):
                severity = "CRITICAL"
            elif act_up in ("JOB_ABORT", "CANCEL", "AUTH_FAIL"):
                severity = "WARNING"
            else:
                severity = "INFO"

        severity = (severity or "INFO")[:16].upper()

        # Automatic duration calculation if request has start_time and duration_ms was not explicitly provided
        if duration_ms is None and request and hasattr(request, "state") and hasattr(request.state, "start_time"):
            try:
                import time
                duration_ms = (time.time() - float(request.state.start_time)) * 1000
            except Exception:
                pass

        duration_val = None
        if duration_ms is not None:
            try:
                duration_val = round(float(duration_ms), 2)
            except (ValueError, TypeError):
                duration_val = None

        # Serialize details dictionary
        details_str = None
        if details:
            if isinstance(details, dict):
                try:
                    details_str = json.dumps(details, default=str)
                except Exception:
                    details_str = str(details)
            else:
                details_str = str(details)

        close_session = False
        if db is None:
            db = SessionLocal()
            close_session = True

        try:
            record = AuditLog(
                timestamp=datetime.now(timezone.utc),
                actor_ip=actor_ip,
                actor_name=actor_name,
                actor_user_id=actor_user_id[:64] if actor_user_id else None,
                action=action[:64].upper(),
                resource_type=resource_type[:64],
                resource_id=str(resource_id)[:128] if resource_id else None,
                status=status[:16].upper(),
                severity=severity,
                request_id=request_id[:64],
                request_method=request_method[:16].upper() if request_method else None,
                endpoint=endpoint[:256] if endpoint else None,
                user_agent=user_agent[:256] if user_agent else None,
                duration_ms=duration_val,
                details=details_str,
            )
            db.add(record)
            db.commit()
            db.refresh(record)
            logger.info(
                "[AUDIT][%s] %s on %s:%s by %s (%s) [%s] req=%s lat=%sms",
                severity,
                action,
                resource_type,
                resource_id or "-",
                actor_name,
                actor_ip,
                status,
                request_id,
                duration_val if duration_val is not None else "-",
            )
            return record.id
        except Exception as e:
            logger.warning("[AUDIT ERROR] Failed to record audit log: %s", e)
            if db:
                try:
                    db.rollback()
                except Exception:
                    pass
            return None
        finally:
            if close_session and db:
                db.close()

    def list_logs(
        self,
        limit: int = 100,
        offset: int = 0,
        action: Optional[str] = None,
        resource_type: Optional[str] = None,
        severity: Optional[str] = None,
        db: Optional[Session] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve recent audit events with optional filtering."""
        if not HAS_SQLALCHEMY or not SessionLocal:
            return []

        close_session = False
        if db is None:
            db = SessionLocal()
            close_session = True

        try:
            query = db.query(AuditLog)
            if action and action.strip():
                query = query.filter(AuditLog.action == action.strip().upper())
            if resource_type and resource_type.strip():
                query = query.filter(AuditLog.resource_type == resource_type.strip().lower())
            if severity and severity.strip():
                query = query.filter(AuditLog.severity == severity.strip().upper())

            records = (
                query.order_by(AuditLog.timestamp.desc())
                .offset(offset)
                .limit(min(limit, 500))
                .all()
            )

            result = []
            for r in records:
                parsed_details = None
                if r.details:
                    try:
                        parsed_details = json.loads(r.details)
                    except Exception:
                        parsed_details = r.details

                result.append(
                    {
                        "id": r.id,
                        "timestamp": r.timestamp.isoformat() if r.timestamp else None,
                        "actor_ip": r.actor_ip,
                        "actor_name": r.actor_name,
                        "actor_user_id": r.actor_user_id,
                        "action": r.action,
                        "resource_type": r.resource_type,
                        "resource_id": r.resource_id,
                        "status": r.status,
                        "severity": r.severity or "INFO",
                        "request_id": r.request_id,
                        "request_method": r.request_method,
                        "endpoint": r.endpoint,
                        "user_agent": r.user_agent,
                        "duration_ms": r.duration_ms,
                        "details": parsed_details,
                    }
                )
            return result
        except Exception as e:
            logger.warning("Failed to query audit logs: %s", e)
            return []
        finally:
            if close_session and db:
                db.close()


audit_service = AuditService()
