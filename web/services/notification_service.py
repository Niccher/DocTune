"""
web/services/notification_service.py
------------------------------------
Dispatches webhook alerts (Slack, Discord, generic HTTP POST) and status notifications
upon job start, completion, failure, and regression detection.
"""

import os
import json
import logging
from typing import Dict, Any, Optional
import httpx

logger = logging.getLogger("doctune.notification")


class NotificationService:
    """Dispatches asynchronous webhooks to team channels."""

    def __init__(self):
        self.webhook_url = os.getenv("WEBHOOK_URL")
        self.slack_url = os.getenv("SLACK_WEBHOOK_URL")
        self.discord_url = os.getenv("DISCORD_WEBHOOK_URL")

    async def notify_job_status(
        self,
        job_id: str,
        job_type: str,
        status: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Send notification to configured webhooks."""
        details = details or {}
        title = f"DocTune Job {job_id} ({job_type}): {status.upper()}"
        color = 0x28A745 if status == "COMPLETED" else (0xDC3545 if status == "FAILED" else 0x007BFF)

        # Standard payload
        payload = {
            "job_id": job_id,
            "job_type": job_type,
            "status": status,
            "details": details,
        }

        # 1. Generic Webhook
        if self.webhook_url:
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    await client.post(self.webhook_url, json=payload)
            except Exception as e:
                logger.warning("Failed to dispatch generic webhook: %s", e)

        # 2. Slack Webhook
        if self.slack_url:
            try:
                slack_body = {
                    "text": f"*{title}*\n" + "\n".join(f">• {k}:* {v}" for k, v in list(details.items())[:5])
                }
                async with httpx.AsyncClient(timeout=5.0) as client:
                    await client.post(self.slack_url, json=slack_body)
            except Exception as e:
                logger.warning("Failed to dispatch Slack webhook: %s", e)

        # 3. Discord Webhook
        if self.discord_url:
            try:
                discord_body = {
                    "embeds": [
                        {
                            "title": title,
                            "color": color,
                            "fields": [{"name": k, "value": str(v), "inline": True} for k, v in list(details.items())[:6]],
                        }
                    ]
                }
                async with httpx.AsyncClient(timeout=5.0) as client:
                    await client.post(self.discord_url, json=discord_body)
            except Exception as e:
                logger.warning("Failed to dispatch Discord webhook: %s", e)


notification_service = NotificationService()
