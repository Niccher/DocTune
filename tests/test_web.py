"""
tests/test_web.py
-----------------
Automated test suite verifying route rendering, APIs, and state.
"""

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


if __name__ == "__main__":
    print("Running tests directly...")
    test_healthcheck_api()
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
    test_strict_no_legacy_fallback()
    test_data_prep_start_cli_args()
    test_normalize_base_url()
    test_extract_json_object_resilience()
    test_storage_service_ensure_buckets()
    test_admin_settings_port_80()
    print("All tests passed successfully!")





