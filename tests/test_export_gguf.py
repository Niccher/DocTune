"""
tests/test_export_gguf.py
-------------------------
Automated test suite verifying the One-Click GGUF & Ollama Exporter:
1. Adapter vs merged model resolution & chat template detection.
2. End-to-end automated export execution (GGUF, Modelfile, run_ollama.sh).
3. Checkpoint database persistence with adapter_type='gguf'.
4. Security and path traversal hardening on /api/v1/models/download-file.
5. API endpoints: POST /api/v1/models/export-one-click & GET /api/v1/models/export-status/{run_id}.
"""

import os
import sys
import json
import shutil
import tempfile
import unittest
from pathlib import Path

# Disable mandatory auth by default for test client
os.environ["AUTH_ENABLED"] = "false"
os.environ["DOCTUNE_ENV"] = "testing"

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from web.main import app
from web.services.export_service import export_service
from web.db.session import init_db, SessionLocal
from web.db.models import Checkpoint

client = TestClient(app)


class TestExportGGUF(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Ensure database is initialized."""
        init_db()

    def test_resolve_adapter_with_custom_base(self):
        """Test that adapter_config.json base_model_name_or_path is detected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_dir = Path(tmpdir)
            cfg_file = adapter_dir / "adapter_config.json"
            with open(cfg_file, "w", encoding="utf-8") as f:
                json.dump({"base_model_name_or_path": "meta-llama/Llama-3.2-3B-Instruct"}, f)

            res = export_service.resolve_adapter_and_base(model_dir=str(adapter_dir))
            self.assertTrue(res["is_adapter"])
            self.assertEqual(res["base_model_id"], "meta-llama/Llama-3.2-3B-Instruct")
            self.assertEqual(res["source_dir"], adapter_dir)

    def test_resolve_merged_model_dir(self):
        """Test that a directory containing safetensors without adapter_config is identified as merged."""
        with tempfile.TemporaryDirectory() as tmpdir:
            merged_dir = Path(tmpdir)
            (merged_dir / "model.safetensors").touch()
            (merged_dir / "config.json").write_text(json.dumps({"model_type": "llama"}), encoding="utf-8")

            res = export_service.resolve_adapter_and_base(model_dir=str(merged_dir))
            self.assertFalse(res["is_adapter"])
            self.assertEqual(res["source_dir"], merged_dir)

    def test_export_one_click_execution(self):
        """Verify one-click export generates GGUF bundle, Modelfile, and registers in DB."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            (model_dir / "model.safetensors").touch()
            (model_dir / "config.json").write_text(json.dumps({"model_type": "qwen2"}), encoding="utf-8")

            export_name = "test-export-qwen"
            res = export_service.export_one_click(
                model_dir=str(model_dir),
                model_name=export_name,
                quantization="q4_k_m",
            )

            self.assertEqual(res["status"], "success")
            self.assertEqual(res["model_name"], export_name)
            self.assertEqual(res["quantization"], "q4_k_m")
            self.assertEqual(res["gguf_filename"], f"{export_name}-q4_k_m.gguf")
            self.assertTrue(Path(res["gguf_path"]).exists())
            self.assertTrue(Path(res["modelfile_path"]).exists())
            self.assertTrue(Path(res["run_script_path"]).exists())

            # Check Modelfile content has qwen chat template
            content = res["modelfile_content"]
            self.assertIn(f"FROM ./{export_name}-q4_k_m.gguf", content)
            self.assertIn("<|im_start|>", content)

            # Check copyable CLI commands
            self.assertIn("ollama create test-export-qwen -f Modelfile", res["ollama_command"])
            self.assertIn("llama-server -m ./test-export-qwen-q4_k_m.gguf", res["llamacpp_command"])

            # Check DB registration
            db = SessionLocal()
            ckpt = db.query(Checkpoint).filter(Checkpoint.disk_path == str(res["gguf_path"])).first()
            self.assertIsNotNone(ckpt)
            self.assertEqual(ckpt.adapter_type, "gguf")
            db.close()

    def test_api_export_one_click_and_download(self):
        """Test API POST /api/v1/models/export-one-click and GET /api/v1/models/download-file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            (model_dir / "model.safetensors").touch()
            (model_dir / "config.json").write_text(json.dumps({"model_type": "llama"}), encoding="utf-8")

            # 1. Trigger export via API
            response = client.post(
                "/api/v1/models/export-one-click",
                data={
                    "model_dir": str(model_dir),
                    "model_name": "api-test-model",
                    "quantization": "q5_k_m",
                },
            )
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["status"], "success")
            self.assertEqual(data["quantization"], "q5_k_m")
            self.assertEqual(data["gguf_filename"], "api-test-model-q5_k_m.gguf")

            # 2. Download generated GGUF file
            dl_res = client.get(f"/api/v1/models/download-file?path={data['gguf_filename']}&model_name=api-test-model")
            self.assertEqual(dl_res.status_code, 200)
            self.assertEqual(dl_res.headers["content-type"], "application/octet-stream")

            # 3. Download Modelfile
            mf_res = client.get("/api/v1/models/download-file?path=Modelfile&model_name=api-test-model")
            self.assertEqual(mf_res.status_code, 200)
            self.assertIn("FROM ./api-test-model-q5_k_m.gguf", mf_res.text)

    def test_download_security_path_traversal_blocked(self):
        """Verify path traversal attacks are rejected with HTTP 400 or 403."""
        # Attempt directory traversal with ..
        res1 = client.get("/api/v1/models/download-file?path=../../etc/passwd")
        self.assertEqual(res1.status_code, 400)

        # Attempt absolute path traversal outside project root
        res2 = client.get("/api/v1/models/download-file?path=/etc/shadow")
        self.assertEqual(res2.status_code, 400)

        # Nonexistent file returns 404
        res3 = client.get("/api/v1/models/download-file?path=nonexistent_model_file_12345.gguf")
        self.assertEqual(res3.status_code, 404)

    def test_get_export_status_endpoint(self):
        """Test GET /api/v1/models/export-status/{run_id}."""
        test_run_id = "test_run_gguf_999"
        export_dir = PROJECT_ROOT / "artifacts" / "runs" / test_run_id / "gguf"
        export_dir.mkdir(parents=True, exist_ok=True)
        try:
            (export_dir / f"{test_run_id}-q4_k_m.gguf").write_bytes(b"GGUF_TEST_BYTES")
            (export_dir / "Modelfile").write_text("FROM ./test.gguf\n", encoding="utf-8")

            res = client.get(f"/api/v1/models/export-status/{test_run_id}")
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertEqual(data["status"], "success")
            self.assertEqual(data["run_id"], test_run_id)
            self.assertEqual(data["gguf_filename"], f"{test_run_id}-q4_k_m.gguf")
            self.assertIn("ollama create", data["ollama_command"])

            # Status for non-existent run returns 404
            res_404 = client.get("/api/v1/models/export-status/nonexistent_run_8888")
            self.assertEqual(res_404.status_code, 404)
        finally:
            shutil.rmtree(PROJECT_ROOT / "artifacts" / "runs" / test_run_id, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
