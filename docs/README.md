# DocTune Studio — Engineering Documentation

Welcome to the engineering documentation hub for **DocTune Studio**. This documentation is designed for software engineers, ML platform engineers, and DevOps contributors who need to understand, modify, test, and deploy the system.

---

## Navigation Directory

| I want to… | Go here |
| :--- | :--- |
| **Run the system without coding** | [../README.md](../README.md) |
| **Understand the distributed system architecture** | [architecture/overview.md](architecture/overview.md) |
| **Inspect inter-service communication & sequence flows** | [architecture/communication.md](architecture/communication.md) |
| **Review database ERD & MinIO bucket architecture** | [architecture/data-and-storage.md](architecture/data-and-storage.md) |
| **Deploy locally or to Google Cloud Platform (GCP)** | [architecture/deployment.md](architecture/deployment.md) |
| **Review threat model & security controls** | [architecture/threat-model.md](architecture/threat-model.md) |
| **Inspect REST API contracts & OpenAPI specs** | [api/contract.md](api/contract.md) • [api/openapi.yaml](api/openapi.yaml) |
| **Work on FastAPI Web Studio** | [services/fastapi.md](services/fastapi.md) |
| **Work on Celery ML Worker & GPU execution** | [services/celery-worker.md](services/celery-worker.md) |
| **Work on MinIO Object Storage** | [services/minio-storage.md](services/minio-storage.md) |
| **Set up native development environment** | [engineering/local-development.md](engineering/local-development.md) |
| **Change behavior safely (Task Matrix & Definition of Done)** | [engineering/making-changes.md](engineering/making-changes.md) |
| **Database migrations & schema changes** | [engineering/database.md](engineering/database.md) |
| **Run and write automated tests** | [engineering/testing.md](engineering/testing.md) |
| **Token safety & secret isolation** | [engineering/security.md](engineering/security.md) |
| **Graceful restarts & incident recovery runbooks** | [runbooks/restart.md](runbooks/restart.md) |
| **Database & MinIO backup/restore runbook** | [runbooks/backup-and-restore.md](runbooks/backup-and-restore.md) |
| **Architecture Decision Records (ADRs)** | [adr/0001-distributed-celery-worker-architecture.md](adr/0001-distributed-celery-worker-architecture.md) |
