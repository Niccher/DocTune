# Celery ML Worker Service

The `celery_worker` service handles all distributed, resource-heavy machine learning workflows, including 4-bit QLoRA training, 16-bit LoRA training, full fine-tuning, and document tokenization.

---

## 1. Concurrency & Queue Configuration

* **Worker Concurrency**: Set to `1` (`-c 1`). Fine-tuning 8B-class models consumes all GPU compute and memory bandwidth; running parallel training jobs on a single GPU will cause out-of-memory errors.
* **Prefetch Multiplier**: Set to `1` (`worker_prefetch_multiplier = 1`) to guarantee that unstarted jobs remain in the Redis queue and can be monitored or cancelled.
* **Task Queues**:
  * `training_queue`: High-priority, GPU-bound fine-tuning and model merge jobs.
  * `data_prep_queue`: CPU-bound document chunking and tokenization.

---

## 2. Command-Line Worker Launch

```bash
# Start Celery worker targeting training and data prep queues
celery -A web.celery_app.app worker --loglevel=INFO -c 1 -Q training_queue,data_prep_queue
```
