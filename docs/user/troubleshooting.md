# User Troubleshooting & Incident Recovery

Solutions to common operational issues when starting or operating DocTune Studio.

---

### 1. Port Collision (Address Already in Use)
* **Symptom**: `Error response from daemon: driver failed programming external connectivity on endpoint doctune-web: Bind for 0.0.0.0:4444 failed: port is already allocated`.
* **Fix**: Edit `.env` and change `WEB_PORT=4445` (or another free port). Re-run `bash scripts/deploy.sh`.

### 2. Disk Space Exhaustion (<15 GB Free Disk)
* **Symptom**: Pre-flight inspection aborts with `Insufficient disk space to safely store checkpoints`.
* **Fix**: Clean Docker build cache and unused volumes:
  ```bash
  docker builder prune -af
  docker image prune -af
  ```
  Check free space using `df -h .`.

### 3. Linux Kernel OOM Killer During Training
* **Symptom**: Training job abruptly stops without error messages; `docker compose logs celery_worker` shows `Killed`.
* **Fix**:
  1. Open `http://localhost:4444/admin` and navigate to **Training & Compute**.
  2. Ensure Fine-Tuning Method is set to **QLoRA (4-bit)** (requires only 9–13 GB memory).
  3. Lower **Batch Size** to `1` or `2`.
  4. Ensure `deploy.sh` created a swapfile or manually create one:
     ```bash
     sudo fallocate -l 4G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile
     ```

### 4. Gated Model 401 Unauthorized (`meta-llama/Meta-Llama-3-8B-Instruct`)
* **Symptom**: Celery worker log displays `GatedRepoError: Access to model meta-llama/Meta-Llama-3-8B-Instruct is restricted`.
* **Fix**:
  1. Obtain an access token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
  2. Accept the model license on Hugging Face.
  3. Enter the token in `http://localhost:4444/admin` under **Hugging Face & Models** and click **Test Token**, then save.
