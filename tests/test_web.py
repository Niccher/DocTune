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
    response = client.get("/api/telemetry")
    assert response.status_code == 200
    data = response.json()
    assert "system" in data
    assert "cpu" in data["system"]
    assert "ram" in data["system"]
    assert "disk" in data["system"]
    assert "gpu" in data["system"]
    assert "pipeline" in data
    assert "checkpoints" in data["pipeline"]


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
    response = client.get("/api/dataset/preview?offset=0&limit=5")
    assert response.status_code == 200
    data = response.json()
    assert "total" in data
    assert "samples" in data


def test_playground_chat_api():
    response = client.post("/api/playground/chat", data={"prompt": "What is the primary failover protocol for Facility B?"})
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
    get_res = client.get("/api/admin/settings")
    assert get_res.status_code == 200
    settings = get_res.json()
    assert "server" in settings
    assert "training_defaults" in settings
    assert "huggingface" in settings

    # 2. Update a setting
    payload = {"training_defaults": {"lora": {"r": 32}}}
    save_res = client.post("/api/admin/settings", json=payload)
    assert save_res.status_code == 200
    updated = save_res.json()
    assert updated["status"] == "success"
    assert updated["settings"]["training_defaults"]["lora"]["r"] == 32

    # 3. Reset settings
    reset_res = client.post("/api/admin/reset")
    assert reset_res.status_code == 200
    restored = reset_res.json()
    assert restored["status"] == "success"
    assert restored["settings"]["training_defaults"]["lora"]["r"] == 16


def test_admin_paths_api():
    response = client.post("/api/admin/check-paths", json={})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "paths" in data
    assert "raw_documents_dir" in data["paths"]
    assert data["paths"]["raw_documents_dir"]["exists"] is True


def test_app_name_dynamic_update():
    # 1. Update app_name to ApexTune Studio
    save_res = client.post("/api/admin/settings", json={"server": {"app_name": "ApexTune Studio"}})
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
    save_res = client.post("/api/admin/settings", json=payload)
    assert save_res.status_code == 200
    data = save_res.json()["settings"]["compute_resources"]
    assert data["cpu_threads"] == 6
    assert data["gpu_device"] == "0"
    assert data["gpu_count"] == 1
    assert data["max_ram_gb"] == 12

    # Reset to defaults
    client.post("/api/admin/reset")



def test_hardware_bounds_api():
    response = client.get("/api/admin/hardware-bounds")
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
    bounds_res = client.get("/api/admin/hardware-bounds")
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
    save_res = client.post("/api/admin/settings", json=payload)
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
    client.post("/api/admin/settings", json={
        "huggingface": {"default_model_id": "custom-org/test-model-v1"},
        "training_defaults": {"num_train_epochs": 7, "per_device_train_batch_size": 2}
    })
    
    # Check training route picks it up via Config.load()
    train_res = client.get("/training")
    assert train_res.status_code == 200
    assert "custom-org/test-model-v1" in train_res.text
    
    # Reset back to defaults
    client.post("/api/admin/reset")


def test_healthcheck_api():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["app"] == "DocTune Studio"
    assert data["version"] == "1.0.0"


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
    print("All tests passed successfully!")



