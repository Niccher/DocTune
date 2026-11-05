"""
web/services/redis_subscriber.py
--------------------------------
Asynchronous Redis Pub/Sub subscriber that bridges real-time distributed
telemetry events from Celery workers into FastAPI AppState and WebSockets.
"""

import os
import json
import logging
import asyncio
from typing import Optional

from web.state import state, JobStatus

logger = logging.getLogger("doctune.redis_subscriber")

_subscriber_task: Optional[asyncio.Task] = None


async def _subscriber_loop(redis_url: str):
    try:
        import redis.asyncio as aioredis
    except ImportError:
        logger.debug("redis.asyncio not installed; skipping Redis Pub/Sub subscriber.")
        return

    logger.info("Starting Redis Pub/Sub telemetry subscriber loop on: %s", redis_url)

    while True:
        client = None
        try:
            client = aioredis.from_url(redis_url, decode_responses=True, socket_connect_timeout=3.0)
            pubsub = client.pubsub()
            await pubsub.psubscribe("run:*:stream")
            logger.info("Successfully subscribed to Redis telemetry pattern: run:*:stream")

            async for message in pubsub.listen():
                if message.get("type") == "pmessage":
                    raw_data = message.get("data")
                    if not raw_data or not isinstance(raw_data, str):
                        continue
                    try:
                        payload = json.loads(raw_data)
                        msg_type = payload.get("type")

                        if msg_type == "log":
                            line = payload.get("line") or payload.get("message", "")
                            if line:
                                await state.add_log(line)

                        elif msg_type == "metric":
                            data = payload.get("data", {})
                            step = int(data.get("step", 0))
                            loss = float(data.get("loss", 0.0))
                            epoch = float(data.get("epoch", 0.0))
                            await state.add_loss_point(step, loss, epoch)

                        elif msg_type == "status":
                            st = payload.get("status")
                            msg = payload.get("message")
                            if msg:
                                await state.add_log(msg)
                            if st == "COMPLETED":
                                state.finish_job(JobStatus.COMPLETED)
                            elif st in ("FAILED", "ABORTED"):
                                state.finish_job(JobStatus.FAILED)

                    except Exception as parse_err:
                        logger.debug("Failed parsing telemetry event payload: %s", parse_err)

        except asyncio.CancelledError:
            logger.info("Redis Pub/Sub subscriber task cancelled.")
            break
        except Exception as e:
            logger.debug("Redis subscriber connection dropped (%s). Retrying in 5s...", e)
            await asyncio.sleep(5)
        finally:
            if client:
                try:
                    await client.close()
                except Exception:
                    pass


def start_redis_subscriber() -> Optional[asyncio.Task]:
    """Launch the background Redis subscriber task if configured."""
    global _subscriber_task
    redis_url = os.getenv("REDIS_URL")
    if not redis_url:
        return None

    if _subscriber_task is None or _subscriber_task.done():
        _subscriber_task = asyncio.create_task(_subscriber_loop(redis_url))
    return _subscriber_task


async def stop_redis_subscriber():
    """Cancel and await termination of the Redis subscriber task."""
    global _subscriber_task
    if _subscriber_task and not _subscriber_task.done():
        _subscriber_task.cancel()
        try:
            await _subscriber_task
        except asyncio.CancelledError:
            pass
        _subscriber_task = None
