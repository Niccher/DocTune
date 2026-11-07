"""
tests/test_web.py
-----------------
Automated test suite verifying route rendering, APIs, and state.
"""

import json
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from web.main import app
from web.state import state, JobType, JobStatus

client = TestClient(app)


def test_dashboard_route():
    response = client.get("/")
    assert response.status_code == 200
    assert "Overview Dashboard" in response.text
    assert "AdminLTE" in response.text or "LLM Studio" in response.text


def test_data_prep_route():
    response = client.get("/data-prep")
    assert response.status_code == 200
    assert "Document & Data Studio" in response.text
    assert "Teacher LLM" in response.text


def test_training_route():
    response = client.get("/training")
    assert response.status_code == 200
    assert "Training Studio" in response.text
    assert "QLoRA" in response.text


def test_models_route():
    response = client.get("/models")
    assert response.status_code == 200
    assert "Checkpoints" in response.text


def test_playground_route():
    response = client.get("/playground")
    assert response.status_code == 200
    assert "Evaluation Benchmark" in response.text


def test_telemetry_api():
    # Canonical /api/v1 endpoint
    response = client.get("/api/v1/telemetry")
    assert response.status_code == 200
    data = response.json()
    assert "system" in data
    assert "cpu" in data["system"]
    assert "ram" in data["system"]
    assert "disk" in data["system"]
    assert "gpu" in data["system"]
    assert "pipeline" in data
    assert "checkpoints" in data["pipeline"]

    # Verify strict non-backward compatibility: legacy unversioned /api/ route returns 404
    legacy_response = client.get("/api/telemetry")
    assert legacy_response.status_code == 404


def test_state_lifecycle():
    assert state.active_job is None
    job = state.set_job(JobType.DATA_PREP, pid=12345)
    assert state.active_job is not None
    assert state.active_job.job_type == JobType.DATA_PREP

    state.finish_job(JobStatus.COMPLETED)
    assert state.active_job is None
    assert len(state.recent_jobs) >= 1
    assert state.recent_jobs[0]["status"] == "COMPLETED"


def test_dataset_preview_api():
    response = client.get("/api/v1/dataset/preview?offset=0&limit=5")
    assert response.status_code == 200
    data = response.json()
    assert "total" in data
    assert "samples" in data


def test_playground_chat_api():
    response = client.post("/api/v1/playground/chat", data={"prompt": "What is the primary failover protocol for Facility B?"})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "diesel auxiliary generators" in data["tuned_response"]


def test_admin_route():
    response = client.get("/admin")
    assert response.status_code == 200
    assert "System Administration" in response.text
    assert "Server &amp; Storage" in response.text or "Server & Storage" in response.text
    assert "Hugging Face" in response.text


def test_admin_settings_apis():
    # 1. Get settings
    get_res = client.get("/api/v1/admin/settings")
    assert get_res.status_code == 200
    settings = get_res.json()
    assert "server" in settings
    assert "training_defaults" in settings
    assert "huggingface" in settings

    # 2. Update a setting
    payload = {"training_defaults": {"lora": {"r": 32}}}
    save_res = client.post("/api/v1/admin/settings", json=payload)
    assert save_res.status_code == 200
    updated = save_res.json()
    assert updated["status"] == "success"
    assert updated["settings"]["training_defaults"]["lora"]["r"] == 32

    # 3. Reset settings
    reset_res = client.post("/api/v1/admin/reset")
    assert reset_res.status_code == 200
    restored = reset_res.json()
    assert restored["status"] == "success"
    assert restored["settings"]["training_defaults"]["lora"]["r"] == 16


def test_admin_paths_api():
    response = client.post("/api/v1/admin/check-paths", json={})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "paths" in data
    assert "raw_documents_dir" in data["paths"]
    assert data["paths"]["raw_documents_dir"]["exists"] is True


def test_app_name_dynamic_update():
    # 1. Update app_name to ApexTune Studio
    save_res = client.post("/api/v1/admin/settings", json={"server": {"app_name": "ApexTune Studio"}})
    assert save_res.status_code == 200

    # 2. Verify rendered HTML contains ApexTune Studio in sidebar, navbar, and title
    res_dash = client.get("/")
    assert "ApexTune Studio" in res_dash.text

    res_admin = client.get("/admin")
    assert "ApexTune Studio" in res_admin.text


def test_compute_resources_update():
    # Update compute resources
    payload = {
        "compute_resources": {
            "cpu_threads": 6,
            "max_ram_gb": 12,
            "gpu_device": "0",
            "gpu_count": 1,
            "cuda_memory_fraction": 0.85
        }
    }
    save_res = client.post("/api/v1/admin/settings", json=payload)
    assert save_res.status_code == 200
    data = save_res.json()["settings"]["compute_resources"]
    assert data["cpu_threads"] == 6
    assert data["gpu_device"] == "0"
    assert data["gpu_count"] == 1
    assert data["max_ram_gb"] == 12

    # Reset to defaults
    client.post("/api/v1/admin/reset")



def test_hardware_bounds_api():
    response = client.get("/api/v1/admin/hardware-bounds")
    assert response.status_code == 200
    bounds = response.json()
    assert "host_cores" in bounds
    assert bounds["host_cores"] >= 1
    assert "total_ram_gb" in bounds
    assert "max_safe_ram" in bounds
    assert "free_disk_gb" in bounds
    assert "has_sufficient_disk" in bounds


def test_boundary_clamping_cpu_ram():
    # 1. Fetch live hardware bounds
    bounds_res = client.get("/api/v1/admin/hardware-bounds")
    bounds = bounds_res.json()
    host_cores = bounds["host_cores"]
    max_safe_ram = bounds["max_safe_ram"]

    # 2. Try to exceed host limits (e.g. 999 cores and 9999 GB RAM)
    payload = {
        "compute_resources": {
            "cpu_threads": 999,
            "max_ram_gb": 9999,
            "gpu_count": 99
        }
    }
    save_res = client.post("/api/v1/admin/settings", json=payload)
    assert save_res.status_code == 200
    resp_data = save_res.json()

    # 3. Verify values were safely clamped to physical ceilings
    cr = resp_data["settings"]["compute_resources"]
    assert cr["cpu_threads"] == host_cores
    assert cr["max_ram_gb"] == max_safe_ram
    assert cr["gpu_count"] <= max(1, bounds["detected_gpus"])
    assert "clamped" in resp_data["message"]

def test_config_load_integration():
    # Verify that updating settings persists and reflects across views
    client.post("/api/v1/admin/settings", json={
        "huggingface": {"default_model_id": "custom-org/test-model-v1"},
        "training_defaults": {"num_train_epochs": 7, "per_device_train_batch_size": 2}
    })
    
    # Check training route picks it up via Config.load()
    train_res = client.get("/training")
    assert train_res.status_code == 200
    assert "custom-org/test-model-v1" in train_res.text
    
    # Reset back to defaults
    client.post("/api/v1/admin/reset")


def test_healthcheck_api():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["app"] == "DocTune Studio"
    assert data["version"] == "1.0.0"


def test_app_lifespan_startup_and_shutdown():
    with TestClient(app) as test_client:
        res = test_client.get("/health")
        assert res.status_code == 200
        assert res.json()["status"] == "healthy"


def test_teacher_presets_api():
    res = client.get("/api/v1/teacher/presets")
    assert res.status_code == 200
    data = res.json()
    assert "presets" in data
    presets = data["presets"]
    for provider in ["ollama", "llamacpp", "vllm", "gemini", "grok", "openai", "custom"]:
        assert provider in presets
        assert "effective_base_url" in presets[provider]
        assert "default_model" in presets[provider]


def test_teacher_connection_api_validation():
    # 1. Invalid URL scheme
    res = client.post("/api/v1/teacher/test", json={"provider": "ollama", "base_url": "ftp://bad-url"})
    assert res.status_code == 200
    data = res.json()
    assert data["valid"] is False
    assert "Invalid Base URL" in data["error"]

    # 2. Missing API key for commercial cloud provider (e.g. Gemini)
    res_gemini = client.post("/api/v1/teacher/test", json={
        "provider": "gemini",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key": ""
    })
    assert res_gemini.status_code == 200
    data_g = res_gemini.json()
    assert data_g["valid"] is False
    assert "API Key is required" in data_g["error"]


def test_teacher_connection_api_mock():
    from unittest.mock import patch, AsyncMock
    mock_result = {
        "valid": True,
        "latency_ms": 42,
        "models": ["llama3.1:latest", "mistral:latest"],
        "active_model": "llama3.1:latest",
        "message": "Connection verified successfully!",
    }
    with patch("web.routers.data_prep.teacher_service.test_connection", new_callable=AsyncMock) as mock_test:
        mock_test.return_value = mock_result
        res = client.post("/api/v1/teacher/test", json={
            "provider": "ollama",
            "base_url": "http://localhost:11434/v1",
            "model": "llama3.1:latest"
        })
        assert res.status_code == 200
        data = res.json()
        assert data["valid"] is True
        assert data["latency_ms"] == 42
        assert "llama3.1:latest" in data["models"]


def test_teacher_models_api():
    from unittest.mock import patch, AsyncMock
    with patch("web.routers.data_prep.teacher_service.fetch_available_models", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = ["model-alpha", "model-beta"]
        res = client.get("/api/v1/teacher/models?provider=ollama")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "success"
        assert data["models"] == ["model-alpha", "model-beta"]


def test_teacher_pull_endpoints():
    """Verify POST /api/v1/teacher/pull and GET /api/v1/teacher/pull-status."""
    # 1. Missing model name returns 400
    res_err = client.post("/api/v1/teacher/pull", json={"provider": "ollama", "model": ""})
    assert res_err.status_code == 400
    assert "required" in res_err.json()["message"].lower()

    # 2. Trigger pull
    res_start = client.post("/api/v1/teacher/pull", json={"provider": "ollama", "model": "test-pull-model"})
    assert res_start.status_code == 200
    assert res_start.json()["status"] == "started"

    # 3. Check pull status
    res_status = client.get("/api/v1/teacher/pull-status?provider=ollama&model=test-pull-model")
    assert res_status.status_code == 200
    status_data = res_status.json()
    assert "status" in status_data
    assert "progress" in status_data

    # 4. Verify completed state retrieval
    from web.services.teacher_service import _pull_tasks, teacher_service
    task_key = teacher_service.get_task_key("ollama", "test-pull-model")
    _pull_tasks[task_key] = {"status": "COMPLETED", "progress": 100, "detail": "Ready!", "model": "test-pull-model"}
    res_comp = client.get("/api/v1/teacher/pull-status?provider=ollama&model=test-pull-model")
    assert res_comp.json()["status"] == "COMPLETED"
    assert res_comp.json()["progress"] == 100

    # 5. Verify failed state retrieval
    _pull_tasks[task_key] = {"status": "FAILED", "progress": 0, "error": "Ollama error: model not found", "model": "test-pull-model"}
    res_fail = client.get("/api/v1/teacher/pull-status?provider=ollama&model=test-pull-model")
    assert res_fail.json()["status"] == "FAILED"
    assert "model not found" in res_fail.json()["error"]

    # 6. Verify non-existent task returns IDLE
    res_idle = client.get("/api/v1/teacher/pull-status?provider=ollama&model=non-existent-task")
    assert res_idle.json()["status"] == "IDLE"
    assert res_idle.json()["progress"] == 0


def test_teacher_connection_diagnostics():
    """Verify connection errors return helpful runner start commands."""
    res = client.post("/api/v1/teacher/test", json={
        "provider": "ollama",
        "base_url": "http://127.0.0.1:54321/v1",
        "model": "any-model"
    })
    assert res.status_code == 200
    data = res.json()
    assert data["valid"] is False
    assert "OLLAMA_HOST=0.0.0.0:11434" in data["error"]


def test_strict_no_legacy_fallback():
    """Verify that unversioned /api/* endpoints return 404 (no backward compatibility aliases)."""
    assert client.get("/api/telemetry").status_code == 404
    assert client.get("/api/teacher/presets").status_code == 404
    assert client.get("/api/admin/settings").status_code == 404
    assert client.post("/api/models/merge").status_code == 404
    assert client.post("/api/training/start").status_code == 404
    assert client.post("/api/data-prep/start").status_code == 404


def test_data_prep_start_cli_args():
    """Verify /api/v1/data-prep/start parses teacher parameters and dispatches pipeline script."""
    from unittest.mock import patch, AsyncMock
    with patch("web.routers.data_prep.process_mgr.run_pipeline_script", new_callable=AsyncMock) as mock_run:
        res = client.post("/api/v1/data-prep/start", data={
            "teacher_provider": "gemini",
            "teacher_url": "https://generativelanguage.googleapis.com/v1beta/openai",
            "teacher_model": "gemini-1.5-flash",
            "teacher_api_key": "AIzaSyTestKey123",
            "max_chunks": "4",
            "chunk_size": "800",
            "overlap": "120",
            "workers": "4",
        })
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "started"
        assert mock_run.called
        kwargs = mock_run.call_args.kwargs
        args_list = kwargs["args"]
        assert "--teacher-url" in args_list
        idx_url = args_list.index("--teacher-url")
        assert args_list[idx_url + 1] == "https://generativelanguage.googleapis.com/v1beta/openai"
        assert "--teacher-model" in args_list
        idx_m = args_list.index("--teacher-model")
        assert args_list[idx_m + 1] == "gemini-1.5-flash"
        assert "--teacher-api-key" in args_list
        idx_k = args_list.index("--teacher-api-key")
        assert args_list[idx_k + 1] == "AIzaSyTestKey123"

        # Clean up active_job in state
        state.active_job = None


def test_normalize_base_url():
    from web.services.teacher_service import teacher_service
    # Ollama without /v1 should automatically gain /v1
    assert teacher_service.normalize_base_url("ollama", "http://localhost:11434") == "http://localhost:11434/v1"
    assert teacher_service.normalize_base_url("ollama", "http://localhost:11434/v1") == "http://localhost:11434/v1"
    assert teacher_service.normalize_base_url("llamacpp", "http://localhost:8080/") == "http://localhost:8080/v1"
    assert teacher_service.normalize_base_url("openai", "https://api.openai.com") == "https://api.openai.com/v1"
    assert teacher_service.normalize_base_url("gemini", "https://generativelanguage.googleapis.com/v1beta") == "https://generativelanguage.googleapis.com/v1beta/openai"


def test_extract_json_object_resilience():
    from pipeline.prepare_data import extract_json_object
    # 1. Clean JSON
    res1 = extract_json_object('{"question": "Q1", "answer": "A1"}')
    assert res1 == {"question": "Q1", "answer": "A1"}

    # 2. Markdown wrapped
    res2 = extract_json_object('```json\n{"question": "Q2", "answer": "A2"}\n```')
    assert res2 == {"question": "Q2", "answer": "A2"}

    # 3. Conversational preamble
    res3 = extract_json_object('Here is your instruction pair:\n{"question": "Q3", "answer": "A3"}\nDone!')
    assert res3 == {"question": "Q3", "answer": "A3"}


def test_storage_service_ensure_buckets():
    from web.services.storage_service import storage_service
    buckets = storage_service.ensure_buckets_exist()
    assert len(buckets) == 4
    assert storage_service.bucket_docs in buckets
    assert storage_service.bucket_datasets in buckets
    assert storage_service.bucket_checkpoints in buckets
    assert storage_service.bucket_merged in buckets


def test_admin_settings_port_80():
    from web.services.config_service import config_service
    settings = config_service.get_settings()
    assert settings["server"]["port"] == 80
    ok, msg, updated = config_service.save_settings({"server": {"port": 80}})
    assert ok is True
    assert updated["server"]["port"] == 80


def test_rehydrate_from_db():
    from web.db.session import SessionLocal, init_db
    from web.db.models import Run, MetricPoint
    init_db()
    from datetime import datetime, timezone
    with SessionLocal() as db:
        db.query(MetricPoint).filter(MetricPoint.run_id == "test_rehydrate_run").delete()
        db.query(Run).filter(Run.id == "test_rehydrate_run").delete()
        db.commit()
        test_run = Run(
            id="test_rehydrate_run",
            job_type="TRAINING_QLORA",
            method="qlora",
            model_id="test/model",
            status="COMPLETED",
            final_loss=1.234,
            created_at=datetime.now(timezone.utc),
        )
        db.add(test_run)
        test_metric = MetricPoint(
            run_id="test_rehydrate_run",
            step=10,
            loss=1.234,
            learning_rate=0.0002,
        )
        db.add(test_metric)
        db.commit()

    state.rehydrate_from_db()
    assert any(j.get("run_id") == "test_rehydrate_run" for j in state.recent_jobs)
    assert any(pt.get("step") == 10 and pt.get("loss") == 1.234 for pt in state.loss_history)


def test_system_settings_db_persistence():
    from web.services.config_service import config_service
    from web.db.session import SessionLocal
    from web.db.models import SystemSetting

    ok, msg, updated = config_service.save_settings({"server": {"app_name": "DocTune DB Test"}})
    assert ok is True
    assert updated["server"]["app_name"] == "DocTune DB Test"

    # Verify presence in SystemSetting table
    with SessionLocal() as db:
        setting = db.query(SystemSetting).filter(SystemSetting.key == "server").first()
        assert setting is not None
        assert setting.value_json["app_name"] == "DocTune DB Test"


    # Reset
    config_service.save_settings({"server": {"app_name": "DocTune Studio"}})


def test_dataset_curation_endpoints():
    from web.db.session import SessionLocal, init_db
    from web.db.models import DatasetSample
    init_db()

    # Create a sample
    with SessionLocal() as db:
        sample = DatasetSample(
            id="test_sample_1",
            system_prompt="You are a helpful assistant.",
            user_instruction="What is test?",
            assistant_response="This is a test response.",
            status="pending",
        )
        db.merge(sample)
        db.commit()

    # GET samples
    resp = client.get("/api/v1/dataset/samples?status=pending")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert any(s["id"] == "test_sample_1" for s in data["samples"])

    # PUT update sample
    resp = client.put("/api/v1/dataset/samples/test_sample_1", json={
        "user_instruction": "Updated instruction?",
        "assistant_response": "Updated response.",
        "quality_score": 5,
    })
    assert resp.status_code == 200
    assert resp.json()["sample"]["user_instruction"] == "Updated instruction?"

    # POST status change
    resp = client.post("/api/v1/dataset/samples/test_sample_1/status", json={"status": "approved"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"

    # POST export jsonl (preserve original file content for clean git tree)
    from pipeline.prepare_data import DEFAULT_JSONL
    orig_content = DEFAULT_JSONL.read_text(encoding="utf-8") if DEFAULT_JSONL.exists() else None
    try:
        resp = client.post("/api/v1/dataset/export-jsonl")
        assert resp.status_code == 200
        export_data = resp.json()
        assert export_data["status"] == "success"
        assert "exported_count" in export_data
    finally:
        if orig_content is not None:
            DEFAULT_JSONL.write_text(orig_content, encoding="utf-8")



def test_playground_feedback_and_history():
    from web.db.session import SessionLocal, init_db
    from web.db.models import PlaygroundLog
    init_db()

    from datetime import datetime, timezone
    with SessionLocal() as db:
        log = PlaygroundLog(
            id="test_play_log_1",
            prompt="Test prompt",
            base_model_response="Base ans",
            tuned_model_response="Tuned ans",
            model_name="test-model",
            created_at=datetime.now(timezone.utc),
        )
        db.merge(log)
        db.commit()

    # Feedback rating
    resp = client.post("/api/v1/playground/feedback", json={
        "log_id": "test_play_log_1",
        "rating": 5,
        "notes": "Excellent accuracy",
    })
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"

    # History API
    resp = client.get("/api/v1/playground/history?limit=100")
    assert resp.status_code == 200
    history = resp.json()
    assert history["status"] == "success"
    assert any(h["id"] == "test_play_log_1" for h in history["history"])


def test_benchmarks_and_checkpoints_api():
    from web.db.session import SessionLocal, init_db
    from web.db.models import EvaluationBenchmark, Checkpoint
    init_db()

    with SessionLocal() as db:
        bench = EvaluationBenchmark(
            id="test_bench_1",
            test_suite_name="Unit Benchmark",
            bleu_score=42.5,
            rouge_l=58.2,
            perplexity=3.14,
        )
        db.merge(bench)
        ckpt = Checkpoint(
            id="test_ckpt_1",
            name="test-ckpt-model",
            adapter_type="qlora",
            step=500,
            val_loss=0.85,
        )
        db.merge(ckpt)
        db.commit()

    # Benchmarks API
    resp = client.get("/api/v1/benchmarks")
    assert resp.status_code == 200
    b_data = resp.json()
    assert b_data["status"] == "success"
    assert any(b["id"] == "test_bench_1" for b in b_data["benchmarks"])

    # Checkpoints API
    resp = client.get("/api/v1/models/checkpoints")
    assert resp.status_code == 200
    c_data = resp.json()
    assert c_data["status"] == "success"
    assert any(c["id"] == "test_ckpt_1" for c in c_data["checkpoints"])


def test_storage_service_directory_and_crud():
    import tempfile
    from web.services.storage_service import storage_service

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        # Create dummy directory structure
        sub_dir = tmp_path / "nested"
        sub_dir.mkdir(parents=True, exist_ok=True)
        (tmp_path / "adapter_config.json").write_text('{"r": 16}', encoding="utf-8")
        (sub_dir / "weights.safetensors").write_bytes(b"dummy_weights_12345")

        # 1. Test upload_dir
        prefix = "unit_test_run_001"
        ok = storage_service.upload_dir(tmp_path, storage_service.bucket_checkpoints, s3_prefix=prefix)
        assert ok is True

        # 2. Test list_objects with prefix
        objs = storage_service.list_objects(storage_service.bucket_checkpoints, prefix=prefix)
        keys = [o["key"] for o in objs]
        assert any("adapter_config.json" in k for k in keys)
        assert any("weights.safetensors" in k for k in keys)

        # 3. Test download_dir
        download_target = tmp_path / "downloaded"
        ok = storage_service.download_dir(storage_service.bucket_checkpoints, prefix, download_target)
        assert ok is True
        assert (download_target / "adapter_config.json").exists()
        assert (download_target / "nested" / "weights.safetensors").exists()
        assert (download_target / "nested" / "weights.safetensors").read_bytes() == b"dummy_weights_12345"

        # 4. Test sync_bucket_to_dir
        sync_target = tmp_path / "synced"
        synced_count = storage_service.sync_bucket_to_dir(storage_service.bucket_checkpoints, sync_target, prefix=prefix)
        assert synced_count >= 2

        # 5. Test delete_dir
        ok = storage_service.delete_dir(storage_service.bucket_checkpoints, prefix=prefix)
        assert ok is True
        objs_after = storage_service.list_objects(storage_service.bucket_checkpoints, prefix=prefix)
        assert len(objs_after) == 0


def test_storage_service_s3_mocked():
    from unittest.mock import MagicMock
    from web.services.storage_service import storage_service
    import tempfile

    old_client = storage_service.s3_client
    mock_s3 = MagicMock()
    storage_service.s3_client = mock_s3

    try:
        assert storage_service.is_s3_enabled() is True

        # 1. ensure_buckets_exist
        mock_s3.list_buckets.return_value = {"Buckets": [{"Name": "doc-tune-raw-docs"}]}
        ensured = storage_service.ensure_buckets_exist()
        assert len(ensured) == 4
        assert mock_s3.create_bucket.call_count >= 3

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            f1 = tmp_path / "model.safetensors"
            f1.write_bytes(b"mock_weights")

            # 2. upload_file & download_file
            assert storage_service.upload_file(f1, "b", "k") is True
            mock_s3.upload_file.assert_called_with(str(f1), "b", "k")

            dl_file = tmp_path / "downloaded.safetensors"
            assert storage_service.download_file("b", "k", dl_file) is True
            mock_s3.download_file.assert_called_with("b", "k", str(dl_file))

            # 3. generate_presigned_download_url
            mock_s3.generate_presigned_url.return_value = "https://s3.example.com/presigned"
            url = storage_service.generate_presigned_download_url("b", "k")
            assert url == "https://s3.example.com/presigned"

            # 4. list_objects with paginator
            mock_paginator = MagicMock()
            mock_paginator.paginate.return_value = [
                {"Contents": [{"Key": "run1/weights.safetensors", "Size": 1024, "LastModified": MagicMock(isoformat=lambda: "2026-11-05T00:00:00")}]}
            ]
            mock_s3.get_paginator.return_value = mock_paginator
            objs = storage_service.list_objects("b", prefix="run1")
            assert len(objs) == 1
            assert objs[0]["key"] == "run1/weights.safetensors"

            # 5. delete_file & delete_dir
            assert storage_service.delete_file("b", "k") is True
            mock_s3.delete_object.assert_called_with(Bucket="b", Key="k")

            assert storage_service.delete_dir("b", prefix="run1") is True
            mock_s3.delete_objects.assert_called_once()

            # 6. upload_dir & download_dir
            assert storage_service.upload_dir(tmp_path, "b", s3_prefix="prefix") is True
            assert storage_service.download_dir("b", "run1", tmp_path / "out") is True
    finally:
        storage_service.s3_client = old_client


def test_document_storage_upload_delete_and_download():
    from io import BytesIO
    from web.db.session import SessionLocal
    from web.db.models import UploadedDocument
    from web.services.storage_service import storage_service

    test_filename = "test_upload_doc.csv"
    file_content = b"col1,col2\nval1,val2\n"

    # 1. Upload via API
    resp = client.post(
        "/api/v1/documents/upload",
        files=[("files", (test_filename, BytesIO(file_content), "text/csv"))],
    )
    assert resp.status_code == 200
    data = resp.json()
    assert test_filename in data["uploaded"]

    # 2. Check Database record has s3_bucket & s3_key
    with SessionLocal() as db:
        doc = db.query(UploadedDocument).filter(UploadedDocument.filename == test_filename).first()
        assert doc is not None
        assert doc.s3_bucket == storage_service.bucket_docs
        assert doc.s3_key == test_filename

    # 3. Download URL endpoint
    resp = client.get(f"/api/v1/documents/{test_filename}/download-url")
    assert resp.status_code == 200
    d_data = resp.json()
    assert d_data["status"] in ["success", "local"]
    assert d_data["filename"] == test_filename

    # 4. Delete endpoint
    resp = client.post("/api/v1/documents/delete", data={"filename": test_filename})
    assert resp.status_code == 200
    assert resp.json()["status"] == "deleted"

    # Verify deleted from DB
    with SessionLocal() as db:
        doc = db.query(UploadedDocument).filter(UploadedDocument.filename == test_filename).first()
        assert doc is None


def test_dataset_storage_export_and_download():
    from pipeline.common import DEFAULT_JSONL
    from web.db.session import SessionLocal
    from web.db.models import DatasetSample

    orig_content = DEFAULT_JSONL.read_text(encoding="utf-8") if DEFAULT_JSONL.exists() else None
    try:
        with SessionLocal() as db:
            sample = DatasetSample(
                id="test_export_sample_1",
                system_prompt="sys",
                user_instruction="What is QLoRA?",
                assistant_response="Quantized Low-Rank Adaptation.",
                status="APPROVED",
            )
            db.merge(sample)
            db.commit()

        # 1. Export JSONL
        resp = client.post("/api/v1/dataset/export-jsonl")
        assert resp.status_code == 200
        exp_data = resp.json()
        assert exp_data["status"] == "success"
        assert exp_data["exported_count"] >= 1
        assert "s3_bucket" in exp_data
        assert exp_data["s3_key"] == "private_docs.jsonl"

        # 2. Download URL endpoint
        resp = client.get("/api/v1/dataset/download-url")
        assert resp.status_code == 200
        dl_data = resp.json()
        assert dl_data["status"] in ["success", "local"]
        assert dl_data["filename"] == "private_docs.jsonl"
    finally:
        if orig_content is not None:
            DEFAULT_JSONL.write_text(orig_content, encoding="utf-8")


def test_checkpoint_s3_endpoints():
    from web.db.session import SessionLocal
    from web.db.models import Checkpoint
    from web.services.storage_service import storage_service
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        (tmp_path / "adapter_config.json").write_text('{"base_model_name_or_path": "meta-llama/Llama-3-8B"}')

        run_id = "test_s3_run_999"
        storage_service.upload_dir(tmp_path, storage_service.bucket_checkpoints, s3_prefix=run_id)

        with SessionLocal() as db:
            ckpt = Checkpoint(
                id=f"ckpt_{run_id}",
                run_id=run_id,
                name="Llama-3-8B-QLORA",
                adapter_type="qlora",
                disk_path=str(tmp_path),
                s3_bucket=storage_service.bucket_checkpoints,
                s3_key=run_id,
                size_mb=12.5,
            )
            db.merge(ckpt)
            db.commit()

        # Test download-url endpoint
        resp = client.get(f"/api/v1/models/checkpoints/ckpt_{run_id}/download-url")
        assert resp.status_code == 200
        d_data = resp.json()
        assert d_data["status"] == "success"
        assert d_data["checkpoint_id"] == f"ckpt_{run_id}"
        assert len(d_data["files"]) >= 1

        # Test pull endpoint
        pull_target = tmp_path / "pulled_local"
        with SessionLocal() as db:
            ckpt_rec = db.query(Checkpoint).filter(Checkpoint.id == f"ckpt_{run_id}").first()
            ckpt_rec.disk_path = str(pull_target)
            db.commit()

        resp = client.post(f"/api/v1/models/checkpoints/ckpt_{run_id}/pull")
        assert resp.status_code == 200
        assert resp.json()["status"] == "success"
        assert (pull_target / "adapter_config.json").exists()


def test_model_indexes_and_timezone_awareness():
    """Verify performance indexes and timezone-aware datetime helpers."""
    from web.db.models import UploadedDocument, DatasetSample, Run, Checkpoint, now_utc

    # 1. Verify indexes
    assert UploadedDocument.file_hash.index is True
    assert DatasetSample.status.index is True
    assert DatasetSample.document_id.index is True
    assert Run.status.index is True
    assert Run.created_at.index is True
    assert Checkpoint.adapter_type.index is True
    assert Checkpoint.created_at.index is True

    # 2. Verify now_utc produces timezone-aware UTC datetime
    dt = now_utc()
    assert dt.tzinfo is not None
    assert str(dt.tzinfo) == "UTC"


def test_alembic_config_and_migrations_exist():
    """Verify Alembic configuration and versioned migration scripts."""
    alembic_ini = PROJECT_ROOT / "alembic.ini"
    env_py = PROJECT_ROOT / "alembic" / "env.py"
    v1 = PROJECT_ROOT / "alembic" / "versions" / "0001_baseline_schema.py"
    v2 = PROJECT_ROOT / "alembic" / "versions" / "0002_add_indexes_and_timezone_support.py"

    assert alembic_ini.exists()
    assert env_py.exists()
    assert v1.exists()
    assert v2.exists()


def test_backup_script_exists_and_executable():
    """Verify database backup script exists and has execution permissions."""
    backup_sh = PROJECT_ROOT / "scripts" / "backup_db.sh"
    assert backup_sh.exists()
    assert os.access(backup_sh, os.X_OK)


def test_resolve_session_semantics():
    """Verify resolve_session correctly differentiates FastAPI DI Depends objects from active Sessions."""
    from web.db.session import resolve_session, SessionLocal
    from fastapi.params import Depends

    # 1. When db is None (default fallback)
    sess1, should_close1 = resolve_session(None)
    assert sess1 is not None
    assert should_close1 is True
    sess1.close()

    # 2. When db is a FastAPI Depends object (direct function call outside request context)
    dummy_depends = Depends(lambda: None)
    sess2, should_close2 = resolve_session(dummy_depends)
    assert sess2 is not None
    assert should_close2 is True
    sess2.close()

    # 3. When db is an actual Session instance
    real_session = SessionLocal()
    sess3, should_close3 = resolve_session(real_session)
    assert sess3 is real_session
    assert should_close3 is False
    real_session.close()


def test_hf_hub_models_list_endpoint():
    """Verify GET /api/v1/models/hub returns catalog with cache and token metadata."""
    resp = client.get("/api/v1/models/hub")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert len(data["models"]) >= 5
    ids = [m["id"] for m in data["models"]]
    assert "Qwen/Qwen2.5-1.5B-Instruct" in ids
    assert "HuggingFaceTB/SmolLM2-1.7B-Instruct" in ids
    for m in data["models"]:
        assert "requires_token" in m
        assert "cached" in m
        assert "params" in m
        assert "is_default" in m


def test_hf_hub_set_default_model_endpoint():
    """Verify POST /api/v1/models/hub/set-default updates default base model."""
    from pipeline.common import Config
    orig_model = Config.load().model_id
    try:
        target_model = "Qwen/Qwen2.5-1.5B-Instruct"
        resp = client.post("/api/v1/models/hub/set-default", data={"model_id": target_model})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["default_model_id"] == target_model

        # Verify Config.load reflects the new model
        cfg = Config.load()
        assert cfg.model_id == target_model
    finally:
        client.post("/api/v1/models/hub/set-default", data={"model_id": orig_model})


def test_training_view_with_query_param():
    """Verify GET /training?model_id=... sets form default from URL."""
    target_model = "microsoft/Phi-3.5-mini-instruct"
    resp = client.get(f"/training?model_id={target_model}")
    assert resp.status_code == 200
    assert target_model in resp.text


def test_models_view_contains_hub_models():
    """Verify GET /models renders the Hugging Face Base Models Hub section."""
    resp = client.get("/models")
    assert resp.status_code == 200
    assert "Hugging Face Base Models Hub" in resp.text
    assert "Qwen 2.5 (1.5B)" in resp.text
    assert "Free (No Token)" in resp.text


def test_hf_hub_download_status_endpoint():
    """Verify GET /api/v1/models/hub/download-status returns status."""
    resp = client.get("/api/v1/models/hub/download-status?model_id=Qwen/Qwen2.5-1.5B-Instruct")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert "cached" in data


def test_hf_hub_search_endpoint():
    """Verify GET /api/v1/models/hub/search works for both empty and queried searches."""
    # Empty query fallback to curated catalog
    resp_empty = client.get("/api/v1/models/hub/search?q=")
    assert resp_empty.status_code == 200
    data_empty = resp_empty.json()
    assert data_empty["status"] == "success"
    assert len(data_empty["models"]) >= 5

    # Filtered query (fallback or mock)
    resp_q = client.get("/api/v1/models/hub/search?q=smol")
    assert resp_q.status_code == 200
    data_q = resp_q.json()
    assert data_q["status"] == "success"
    assert any("smol" in m["id"].lower() or "smol" in m["name"].lower() for m in data_q["models"])


def test_hf_hub_download_trigger_and_progress():
    """Verify POST /api/v1/models/hub/download registers task and download-status reflects it."""
    test_model = "test-org/mock-base-model"
    resp = client.post("/api/v1/models/hub/download", data={"model_id": test_model})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "started"
    assert data["model_id"] == test_model

    # Check status endpoint immediately after trigger (may be STARTING, DOWNLOADING, IDLE or FAILED if offline)
    status_resp = client.get(f"/api/v1/models/hub/download-status?model_id={test_model}")
    assert status_resp.status_code == 200
    status_data = status_resp.json()
    assert status_data["status"] == "success"
    assert status_data["download_status"] in ("STARTING", "DOWNLOADING", "IDLE", "FAILED")


def test_hf_hub_cache_deduplication():
    """Verify hf_model_service.check_model_cache avoids double-counting symlinked blobs."""
    import tempfile
    from pathlib import Path
    from web.services.hf_model_service import hf_model_service

    with tempfile.TemporaryDirectory() as tmp:
        cache_base = Path(tmp)
        folder = cache_base / "models--testorg--testmodel"
        blobs = folder / "blobs"
        blobs.mkdir(parents=True)
        blob = blobs / "sha256_weight"
        blob.write_bytes(b"0" * (1024 * 1024))  # 1 MB

        snaps = folder / "snapshots" / "main"
        snaps.mkdir(parents=True)
        link = snaps / "model.safetensors"
        link.symlink_to(blob)

        # Monkeypatch get_cache_dir
        orig_get_cache = hf_model_service.get_cache_dir
        try:
            hf_model_service.get_cache_dir = lambda: cache_base
            res = hf_model_service.check_model_cache("testorg/testmodel")
            assert res["cached"] is True
            assert res["size_mb"] == 1.0  # Exactly 1.0 MB, NOT 2.0 MB
        finally:
            hf_model_service.get_cache_dir = orig_get_cache


def test_pii_redactor_and_preview_api():
    import json
    from pipeline.pii_redactor import PIIRedactor
    redactor = PIIRedactor(mode="placeholder")
    raw = "Contact Alice at alice@company.com or 415-555-9876. SSN: 000-11-2222. API Key: sk-prod12345678901234567890."
    clean, counts = redactor.redact(raw)
    assert "[EMAIL_ADDRESS]" in clean
    assert "alice@company.com" not in clean
    assert "000-11-2222" not in clean
    assert "sk-prod12345678901234567890" not in clean
    assert counts.get("EMAIL_ADDRESS", 0) >= 1
    assert counts.get("US_SSN", 0) >= 1

    # Test preview API endpoint
    resp = client.post("/api/v1/data-prep/preview-redaction", json={"text": raw, "mode": "placeholder"})
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["status"] == "success"
    assert "alice@company.com" not in data["redacted_text"]
    assert len(data["entities_detected"]) >= 2


def test_gguf_export_and_modelfile_api():
    import json
    import shutil
    from pipeline.export_gguf import detect_chat_template, generate_ollama_modelfile, run_conversion

    mock_dir = PROJECT_ROOT / "data" / "mock_llama_model"
    mock_dir.mkdir(parents=True, exist_ok=True)
    try:
        (mock_dir / "config.json").write_text(json.dumps({"model_type": "llama"}), encoding="utf-8")
        tmpl = detect_chat_template(mock_dir)
        assert "<|start_header_id|>" in tmpl["template"]
        mf = generate_ollama_modelfile("model.gguf", tmpl)
        assert "FROM ./model.gguf" in mf
        assert "<|eot_id|>" in mf

        # Test Modelfile endpoint
        resp = client.get("/api/v1/models/modelfile?model_name=test-model")
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["status"] == "success"
        assert "ollama create" in data["run_command"]

        # Test export endpoint
        out_gguf_dir = PROJECT_ROOT / "data" / "test_out_gguf_api"
        resp_export = client.post("/api/v1/models/export-gguf", data={
            "model_dir": str(mock_dir),
            "export_dir": str(out_gguf_dir),
            "model_name": "test-model-api",
            "quantization": "q4_k_m",
        })
        assert resp_export.status_code in [200, 409]
    finally:
        shutil.rmtree(mock_dir, ignore_errors=True)
        shutil.rmtree(PROJECT_ROOT / "data" / "test_out_gguf_api", ignore_errors=True)


def test_openai_compatible_inference_api():
    # 1. /v1/models
    res = client.get("/v1/models")
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["object"] == "list"
    assert len(body["data"]) > 0

    # 2. /v1/chat/completions non-streaming
    payload = {
        "model": "doctune-merged",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the primary failover protocol for Facility B?"}
        ]
    }
    res2 = client.post("/v1/chat/completions", json=payload)
    assert res2.status_code == 200, res2.text
    data2 = res2.json()
    assert data2["object"] == "chat.completion"
    assert "diesel auxiliary generators" in data2["choices"][0]["message"]["content"]
    assert data2["usage"]["total_tokens"] > 0

    # 3. /v1/chat/completions streaming
    payload["stream"] = True
    res3 = client.post("/v1/chat/completions", json=payload)
    assert res3.status_code == 200, res3.text
    assert "text/event-stream" in res3.headers["content-type"]
    assert "data: [DONE]" in res3.text


def test_playground_dpo_export_and_stats():
    import json
    from web.db.session import SessionLocal
    from web.db.models import PlaygroundLog

    # Seed playground logs with ratings
    db = SessionLocal() if SessionLocal else None
    if db:
        try:
            db.query(PlaygroundLog).filter(PlaygroundLog.id.in_(["test_dpo_1", "test_dpo_2"])).delete()
            db.add(PlaygroundLog(
                id="test_dpo_1",
                prompt="What is protocol X?",
                base_model_response="Base generic answer",
                tuned_model_response="Fine-tuned grounded answer",
                rating="thumbs_up",
            ))
            db.add(PlaygroundLog(
                id="test_dpo_2",
                prompt="Who is the director?",
                base_model_response="Unknown",
                tuned_model_response="Hallucinated answer",
                rating="thumbs_down",
                notes="The actual director is Alice Smith",
            ))
            db.commit()
        finally:
            db.close()

    # Test DPO stats endpoint
    res_stats = client.get("/api/v1/playground/dpo/stats")
    assert res_stats.status_code == 200, res_stats.text
    stats = res_stats.json()
    assert stats["status"] == "success"
    assert stats["thumbs_up"] >= 1
    assert stats["thumbs_down"] >= 1

    # Test DPO export endpoint
    res_export = client.post("/api/v1/playground/dpo/export")
    assert res_export.status_code == 200, res_export.text
    exp_data = res_export.json()
    assert exp_data["status"] == "success"
    assert exp_data["exported_count"] >= 2
    pref_path = Path(exp_data["path"])
    assert pref_path.exists()

    # Validate JSONL contents
    with open(pref_path, "r", encoding="utf-8") as f:
        lines = [json.loads(l) for l in f]
    assert any(l["chosen"] == "Fine-tuned grounded answer" and l["rejected"] == "Base generic answer" for l in lines)
    assert any(l["chosen"] == "The actual director is Alice Smith" and l["rejected"] == "Hallucinated answer" for l in lines)


def test_dataset_versioning_api():
    """Verify immutable dataset snapshot creation, listing, and activation."""
    from web.services.dataset_version_service import dataset_version_service
    from pipeline.common import DEFAULT_JSONL

    orig_content = DEFAULT_JSONL.read_text(encoding="utf-8") if DEFAULT_JSONL.exists() else None
    test_file = DEFAULT_JSONL.parent / "test_snapshot_src.jsonl"
    test_file.write_text(json.dumps({"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}) + "\n")
    entry = None
    try:
        entry = dataset_version_service.create_version_from_file(test_file, tag="test-v99", description="Unit test snapshot")
        assert entry["version_tag"] == "test-v99"
        assert entry["sample_count"] == 1
        assert len(entry["sha256"]) == 64

        # List versions API
        res = client.get("/api/v1/dataset/versions")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "success"
        assert any(v["version_id"] == entry["version_id"] for v in data["versions"])

        # Activate version API
        res_act = client.post("/api/v1/dataset/versions/activate", json={"version_id": entry["version_id"]})
        assert res_act.status_code == 200
        assert res_act.json()["status"] == "success"
    finally:
        if test_file.exists():
            test_file.unlink()
        if entry:
            p = Path(entry["path"])
            if p.exists():
                p.unlink()
            dataset_version_service.delete_version(entry["version_id"])
        if orig_content is not None:
            DEFAULT_JSONL.write_text(orig_content, encoding="utf-8")


def test_training_queue_and_resumable_checkpoints():
    """Verify queue management, FIFO ordering, and checkpoint discovery."""
    from web.state import state, JobInfo, JobType, JobStatus
    from web.services.queue_manager import queue_manager

    # Checkpoint discovery API
    res_chk = client.get("/api/v1/training/checkpoints/resumable")
    assert res_chk.status_code == 200
    assert res_chk.json()["status"] == "success"

    # Fake an active job in state to force enqueueing
    fake_job = JobInfo(job_id="busy_job", job_type=JobType.TRAINING_QLORA, status=JobStatus.RUNNING)
    old_active = state.active_job
    state.active_job = fake_job

    try:
        res = client.post("/api/v1/training/start", data={
            "method": "qlora",
            "model_id": "meta-llama/Meta-Llama-3-8B-Instruct",
            "epochs": 2,
            "batch_size": 2,
            "grad_accum": 4,
            "lora_r": 16,
            "lora_alpha": 32,
            "learning_rate": 0.0002,
            "max_seq_length": 512,
            "resume_checkpoint": "auto",
        })
        assert res.status_code == 200
        q_data = res.json()
        assert q_data["status"] == "queued"
        run_id = q_data["job_id"]

        # Check queue API
        res_q = client.get("/api/v1/training/queue")
        assert res_q.status_code == 200
        assert any(j["run_id"] == run_id for j in res_q.json()["queue"])

        # Cancel queued job
        res_del = client.delete(f"/api/v1/training/queue/{run_id}")
        assert res_del.status_code == 200
        assert res_del.json()["status"] == "success"
    finally:
        state.active_job = old_active


def test_hyperparameter_sweep_generation():
    """Verify combinatorial hyperparameter grid generation and queue population."""
    from web.services.queue_manager import queue_manager

    res = client.post("/api/v1/training/sweeps/generate", json={
        "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "method": "qlora",
        "learning_rates": "0.0001, 0.0002",
        "lora_ranks": "16, 32",
        "epochs_list": "2",
    })
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert data["total_trials"] == 4  # 2 LRs * 2 Ranks * 1 Epoch = 4
    assert len(data["trials"]) == 4

    for t in data["trials"]:
        queue_manager.remove(t["run_id"])


def test_experiments_run_comparison_api():
    """Verify multi-run comparison endpoint and telemetry alignment."""
    res_html = client.get("/experiments")
    assert res_html.status_code == 200
    assert "Run Comparison" in res_html.text

    res_runs = client.get("/api/v1/experiments/runs")
    assert res_runs.status_code == 200
    assert res_runs.json()["status"] == "success"

    res_cmp = client.get("/api/v1/experiments/compare?run_ids=mock_run_1,mock_run_2")
    assert res_cmp.status_code == 200
    cmp_data = res_cmp.json()
    assert cmp_data["status"] == "success"
    assert len(cmp_data["series"]) >= 2
    assert len(cmp_data["table"]) >= 2


def test_huggingface_push_to_hub_api():
    """Verify push to hub validation and security checks."""
    res_err = client.post("/api/v1/models/push-to-hub", json={"model_name": "merged_model", "repo_id": ""})
    assert res_err.status_code == 400

    res_missing = client.post("/api/v1/models/push-to-hub", json={
        "model_name": "non_existent_model_xyz",
        "repo_id": "org/test-model",
    })
    assert res_missing.status_code == 404


def test_rag_benchmark_engine():
    """Verify 4-way empirical RAG baseline vs fine-tuning benchmark."""
    res = client.get("/api/v1/experiments/rag-benchmark")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert "summary" in data
    summary = data["summary"]

    assert "base_zero_shot" in summary
    assert "base_rag" in summary
    assert "finetuned_model" in summary
    assert "finetuned_rag" in summary

    assert "factual_accuracy_pct" in summary["finetuned_model"]
    assert "avg_latency_ms" in summary["finetuned_model"]
    assert "avg_tokens" in summary["finetuned_model"]
    assert len(data["details"]) >= 5


if __name__ == "__main__":
    print("Running tests directly...")
    test_healthcheck_api()
    test_app_lifespan_startup_and_shutdown()
    test_dashboard_route()
    test_data_prep_route()
    test_training_route()
    test_models_route()
    test_playground_route()
    test_admin_route()
    test_telemetry_api()
    test_admin_settings_apis()
    test_admin_paths_api()
    test_app_name_dynamic_update()
    test_compute_resources_update()
    test_hardware_bounds_api()
    test_boundary_clamping_cpu_ram()
    test_config_load_integration()
    test_dataset_preview_api()
    test_playground_chat_api()
    test_state_lifecycle()
    test_teacher_presets_api()
    test_teacher_connection_api_validation()
    test_teacher_connection_api_mock()
    test_teacher_models_api()
    test_teacher_pull_endpoints()
    test_teacher_connection_diagnostics()
    test_strict_no_legacy_fallback()
    test_data_prep_start_cli_args()
    test_normalize_base_url()
    test_extract_json_object_resilience()
    test_storage_service_ensure_buckets()
    test_storage_service_directory_and_crud()
    test_storage_service_s3_mocked()
    test_document_storage_upload_delete_and_download()
    test_dataset_storage_export_and_download()
    test_checkpoint_s3_endpoints()
    test_admin_settings_port_80()
    test_rehydrate_from_db()
    test_system_settings_db_persistence()
    test_dataset_curation_endpoints()
    test_playground_feedback_and_history()
    test_benchmarks_and_checkpoints_api()
    test_hf_hub_models_list_endpoint()
    test_hf_hub_set_default_model_endpoint()
    test_training_view_with_query_param()
    test_models_view_contains_hub_models()
    test_hf_hub_download_status_endpoint()
    test_hf_hub_search_endpoint()
    test_hf_hub_download_trigger_and_progress()
    test_hf_hub_cache_deduplication()
    test_model_indexes_and_timezone_awareness()
    test_alembic_config_and_migrations_exist()
    test_backup_script_exists_and_executable()
    test_resolve_session_semantics()
    test_pii_redactor_and_preview_api()
    test_gguf_export_and_modelfile_api()
    test_openai_compatible_inference_api()
    test_playground_dpo_export_and_stats()
    test_dataset_versioning_api()
    test_training_queue_and_resumable_checkpoints()
    test_hyperparameter_sweep_generation()
    test_experiments_run_comparison_api()
    test_huggingface_push_to_hub_api()
    test_rag_benchmark_engine()
    print("All tests passed successfully!")







