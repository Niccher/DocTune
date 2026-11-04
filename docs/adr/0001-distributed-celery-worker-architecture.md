# ADR 0001: Distributed Celery Worker Architecture

* **Status**: Accepted
* **Date**: 2026-11-04

## Context and Problem Statement
LLM fine-tuning consumes significant CPU, RAM, and GPU resources. When running training tasks directly inside the FastAPI web process, long-running blocking operations can starve the HTTP event loop, cause request timeouts, and crash the web server when out-of-memory errors occur.

## Decision Drivers
* Isolate web server availability from ML training crashes.
* Support running workers on dedicated GPU compute nodes independently of the web dashboard.
* Enable controlled job serialization (one GPU job at a time).

## Considered Options
1. In-process `asyncio.subprocess` (Previous implementation).
2. Distributed Celery Workers with Redis message broker.
3. Custom multi-threading queue inside FastAPI.

## Decision Outcome
**Option 2: Distributed Celery Workers**.
Celery with Redis provides battle-tested worker lifecycle management, queue routing, retry policies, and clean decoupling between UI interactions and PyTorch execution.
