# ADR 0003: Redis Pub/Sub for Real-Time Telemetry Streaming

* **Status**: Accepted
* **Date**: 2026-11-04

## Context and Problem Statement
Operators require real-time loss curves and live console stdout streaming in the browser during training. In a distributed multi-container setup, the training worker runs in a separate container from the FastAPI web server.

## Decision Drivers
* Real-time delivery with low latency (<50 ms).
* Decoupled producer (Celery Worker) and consumer (FastAPI WebSockets).
* Support for multiple browser sessions observing the same training run simultaneously.

## Decision Outcome
**Redis Pub/Sub Channels**.
Workers publish lines to `run:{run_id}:stream`. The FastAPI WebSocket handler subscribes to the channel and relays messages as WebSocket text frames directly to ApexCharts and the terminal log viewer.
