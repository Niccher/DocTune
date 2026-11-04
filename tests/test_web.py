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


if __name__ == "__main__":
    print("Running tests directly...")
    test_dashboard_route()
    test_data_prep_route()
    test_training_route()
    test_models_route()
    test_playground_route()
    test_telemetry_api()
    test_dataset_preview_api()
    test_playground_chat_api()
    test_state_lifecycle()
    print("All tests passed successfully!")
