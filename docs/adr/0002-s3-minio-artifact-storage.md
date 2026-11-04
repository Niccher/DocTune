# ADR 0002: S3 / MinIO Object Storage for Artifacts

* **Status**: Accepted
* **Date**: 2026-11-04

## Context and Problem Statement
LLM model checkpoints and merged weights range between 150 MB (LoRA adapters) and 16 GB (merged 16-bit models). Storing them in Docker container layers leads to container bloating and disk exhaustion. Storing them in local host paths complicates multi-node clustering and cloud deployment.

## Decision Drivers
* S3 API compatibility across local testing and cloud environments (GCP Cloud Storage, AWS S3).
* Ability to generate time-limited Presigned URLs for direct client downloads without passing gigabytes through FastAPI memory.

## Decision Outcome
**MinIO S3 Object Storage**.
MinIO provides a self-hosted S3-compatible service for local development and Docker Compose, while enabling instant zero-code switching to Google Cloud Storage (GCS) in production via S3 interoperability keys.
