"""
web/services/config_service.py
------------------------------
Service for managing system configurations, JSON persistence,
Hugging Face token validation, directory checks, and webhook alerts.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import urllib.request
import urllib.error

from pipeline.common import PROJECT_ROOT, CONFIG_DIR, CONFIG_FILE
from web.services.system_service import get_hardware_bounds


logger = logging.getLogger(__name__)

FACTORY_DEFAULTS: Dict[str, Any] = {
    "server": {
        "app_name": "DocTune Studio",
        "host": "0.0.0.0",
        "port": 80,
        "log_level": "INFO",
        "max_concurrent_jobs": 1,
        "max_log_buffer_lines": 1000,
        "theme": "dark",
    },
    "compute_resources": {
        "cpu_threads": 4,
        "max_ram_gb": 10,
        "gpu_device": "0",
        "gpu_count": 1,
        "cuda_memory_fraction": 0.90,
    },

    "huggingface": {
        "token": "",
        "cache_dir": "~/.cache/huggingface/hub",
        "default_model_id": "meta-llama/Meta-Llama-3-8B-Instruct",
    },
    "storage_paths": {
        "raw_documents_dir": "raw_documents",
        "data_dir": "data",
        "output_qlora_dir": "output_qlora",
        "output_lora_dir": "output_lora",
        "output_fft_dir": "output_fft",
        "output_merged_dir": "output_merged",
        "runs_dir": ".runs",
    },
    "training_defaults": {
        "num_train_epochs": 3,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "learning_rate_lora": 0.0002,
        "learning_rate_fft": 0.00001,
        "max_seq_length": 1024,
        "logging_steps": 10,
        "seed": 42,
        "precision": "bf16",
        "lora": {
            "r": 16,
            "alpha": 32,
            "dropout": 0.05,
            "target_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        },
    },
    "data_prep": {
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "train_eval_split": 0.9,
        "qa_generator": {
            "enabled": False,
            "provider": "ollama",
            "base_url": "http://localhost:11434/v1",
            "api_key": "",
            "model": "llama3:latest",
            "temperature": 0.3,
        },
    },
    "integrations": {
        "wandb": {
            "enabled": False,
            "api_key": "",
            "project": "llm-private-fine-tuning",
            "entity": "",
        },
        "webhooks": {
            "slack_discord_url": "",
            "notify_on_complete": True,
            "notify_on_failure": True,
        },
    },
}


def _load_settings_from_db() -> Optional[Dict[str, Any]]:
    try:
        from web.db.session import SessionLocal
        from web.db.models import SystemSetting
        if not SessionLocal:
            return None
        db = SessionLocal()
        try:
            records = db.query(SystemSetting).all()
            if not records:
                return None
            result = {}
            for r in records:
                result[r.key] = r.value_json
            return result
        finally:
            db.close()
    except Exception:
        return None


def _save_settings_to_db(settings_dict: Dict[str, Any]):
    try:
        from web.db.session import SessionLocal
        from web.db.models import SystemSetting
        if not SessionLocal:
            return
        db = SessionLocal()
        try:
            for key, val in settings_dict.items():
                existing = db.query(SystemSetting).filter(SystemSetting.key == key).first()
                if existing:
                    existing.value_json = val
                    existing.version = (existing.version or 1) + 1
                else:
                    new_item = SystemSetting(key=key, value_json=val, version=1)
                    db.add(new_item)
            db.commit()
        finally:
            db.close()
    except Exception as e:
        logger.debug("Failed writing settings to DB (using file storage): %s", e)


class ConfigService:
    def __init__(self, config_file: Path = CONFIG_FILE):
        self.config_file = config_file

    def get_settings(self) -> Dict[str, Any]:
        """Read and return current settings from PostgreSQL with disk/defaults fallback."""
        db_settings = _load_settings_from_db()
        if db_settings:
            merged = FACTORY_DEFAULTS.copy()
            for key, val in db_settings.items():
                if isinstance(val, dict) and key in merged and isinstance(merged[key], dict):
                    merged[key] = {**merged[key], **val}
                else:
                    merged[key] = val
            return merged

        if not self.config_file.exists():
            self.reset_to_defaults()
            return FACTORY_DEFAULTS.copy()

        try:
            with open(self.config_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            merged = FACTORY_DEFAULTS.copy()
            for key, val in data.items():
                if isinstance(val, dict) and key in merged and isinstance(merged[key], dict):
                    merged[key] = {**merged[key], **val}
                else:
                    merged[key] = val
            _save_settings_to_db(merged)
            return merged
        except Exception as e:
            logger.error("Error reading config file %s: %s", self.config_file, e)
            return FACTORY_DEFAULTS.copy()

    def save_settings(self, new_settings: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """Validate and write updated settings to disk atomically."""
        try:
            current = self.get_settings()
            bounds = get_hardware_bounds()
            clamped_notices: List[str] = []

            # Deep update with boundary validation and clamping
            for section, values in new_settings.items():
                if section in current and isinstance(values, dict):
                    for k, v in values.items():
                        if section == "server":
                            if k == "app_name":
                                v = str(v).strip() or "DocTune Studio"
                            elif k == "port":
                                v = max(1, min(int(v), 65535))
                            elif k == "max_concurrent_jobs":
                                v = max(1, min(int(v), 4))
                            elif k == "max_log_buffer_lines":
                                v = max(100, min(int(v), 10000))
                        elif section == "compute_resources":
                            if k == "cpu_threads":
                                orig = int(v)
                                v = max(1, min(orig, bounds["host_cores"]))
                                if v != orig:
                                    clamped_notices.append(f"CPU threads clamped to {v} (host has {bounds['host_cores']} cores)")
                            elif k == "max_ram_gb":
                                orig = int(v)
                                v = max(1, min(orig, bounds["max_safe_ram"]))
                                if v != orig:
                                    clamped_notices.append(f"RAM budget clamped to {v} GB (max safe: {bounds['max_safe_ram']} GB)")
                            elif k == "gpu_count":
                                orig = int(v)
                                max_g = max(1, bounds["detected_gpus"])
                                v = max(0, min(orig, max_g))
                                if v != orig:
                                    clamped_notices.append(f"GPU count clamped to {v} (detected: {bounds['detected_gpus']} physical GPUs)")
                            elif k == "cuda_memory_fraction":
                                orig = float(v)
                                v = min(0.95, max(0.1, orig))
                                if v != orig:
                                    clamped_notices.append(f"VRAM fraction clamped to {v}")
                            elif k == "gpu_device":
                                v = str(v).strip()
                        elif section == "training_defaults":
                            if k == "per_device_train_batch_size":
                                v = max(1, min(int(v), 32))
                            elif k == "gradient_accumulation_steps":
                                v = max(1, min(int(v), 64))
                            elif k == "max_seq_length":
                                v = max(128, min(int(v), 8192))
                            elif k == "num_train_epochs":
                                v = max(1, min(int(v), 50))
                            elif k == "logging_steps":
                                v = max(1, min(int(v), 1000))
                            elif k == "seed":
                                v = int(v)
                            elif k in ["learning_rate_lora", "learning_rate_fft"]:
                                v = max(1e-7, min(float(v), 0.1))
                            elif k == "lora" and isinstance(v, dict):
                                if "r" in v:
                                    v["r"] = max(2, min(int(v["r"]), 256))
                                if "alpha" in v:
                                    v["alpha"] = max(2, min(int(v["alpha"]), 512))
                                if "dropout" in v:
                                    v["dropout"] = max(0.0, min(float(v["dropout"]), 0.50))
                                if "target_modules" in v and isinstance(v["target_modules"], str):
                                    v["target_modules"] = [m.strip() for m in v["target_modules"].split(",") if m.strip()]
                        elif section == "data_prep":
                            if k == "chunk_size":
                                v = max(100, min(int(v), 5000))
                            elif k == "chunk_overlap":
                                c_size = current.get("data_prep", {}).get("chunk_size", 1000)
                                v = max(0, min(int(v), int(c_size * 0.5)))
                            elif k == "train_eval_split":
                                v = max(0.5, min(float(v), 0.99))
                            elif k == "qa_generator" and isinstance(v, dict):
                                if "temperature" in v:
                                    v["temperature"] = max(0.0, min(float(v["temperature"]), 1.5))
                        current[section][k] = v
                else:
                    current[section] = values

            # Write atomically to temp then rename
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            temp_file = self.config_file.with_suffix(".tmp")
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(current, f, indent=2)
            temp_file.replace(self.config_file)

            # Persist to database
            _save_settings_to_db(current)

            msg = "Configuration saved successfully"
            if clamped_notices:
                msg += f" (Note: {', '.join(clamped_notices)})"

            return True, msg, current
        except Exception as e:
            logger.error("Failed to save configuration: %s", e)
            return False, f"Failed to save configuration: {str(e)}", self.get_settings()


    def reset_to_defaults(self) -> Dict[str, Any]:
        """Reset settings back to factory defaults in DB and disk."""
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(self.config_file, "w", encoding="utf-8") as f:
            json.dump(FACTORY_DEFAULTS, f, indent=2)
        _save_settings_to_db(FACTORY_DEFAULTS)
        return FACTORY_DEFAULTS.copy()

    def get_app_name(self) -> str:
        """Return the current configured application name."""
        return self.get_settings().get("server", {}).get("app_name", "DocTune Studio")

    def get_app_version(self) -> str:
        """Return the current application version from the VERSION file or fallback."""
        try:
            version_file = PROJECT_ROOT / "VERSION"
            if version_file.is_file():
                v = version_file.read_text(encoding="utf-8").strip()
                if v:
                    return v
        except Exception:
            pass
        return "1.3.0"

    def get_compute_resources(self) -> Dict[str, Any]:
        """Return active compute resource constraints."""
        return self.get_settings().get("compute_resources", FACTORY_DEFAULTS["compute_resources"])


    def test_hf_token(self, token: str) -> Dict[str, Any]:
        """Check token with Hugging Face API."""
        if not token or not token.strip():
            return {"valid": False, "error": "Token string is empty"}

        req = urllib.request.Request(
            "https://huggingface.co/api/whoami-v2",
            headers={"Authorization": f"Bearer {token.strip()}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode("utf-8"))
                    name = data.get("name", "Unknown")
                    orgs = [o.get("name") for o in data.get("orgs", []) if "name" in o]
                    return {
                        "valid": True,
                        "name": name,
                        "orgs": orgs,
                        "message": f"Authenticated successfully as {name} ({', '.join(orgs) if orgs else 'Individual'})",
                    }
                return {"valid": False, "error": f"Unexpected HTTP status {response.status}"}
        except urllib.error.HTTPError as e:
            if e.code == 401:
                return {"valid": False, "error": "Invalid or expired Hugging Face token (HTTP 401 Unauthorized)"}
            return {"valid": False, "error": f"Hugging Face API returned HTTP {e.code}"}
        except Exception as e:
            return {"valid": False, "error": f"Connection error: {str(e)}"}

    def verify_and_create_paths(self, storage_paths: Dict[str, str] = None) -> Dict[str, Any]:
        """Verify directories exist on disk and create them if absent."""
        if storage_paths is None:
            settings = self.get_settings()
            storage_paths = settings.get("storage_paths", {})

        results = {}
        for key, rel_path in storage_paths.items():
            path = PROJECT_ROOT / rel_path
            created = False
            try:
                if not path.exists():
                    path.mkdir(parents=True, exist_ok=True)
                    created = True
                writable = path.is_dir() and Path(path / ".test_write").touch(exist_ok=True) is None
                if (path / ".test_write").exists():
                    (path / ".test_write").unlink()
                results[key] = {
                    "path": str(path),
                    "exists": True,
                    "created": created,
                    "writable": writable,
                }
            except Exception as e:
                results[key] = {
                    "path": str(path),
                    "exists": False,
                    "created": False,
                    "writable": False,
                    "error": str(e),
                }

        return {"status": "ok", "paths": results}

    def test_webhook(self, url: str) -> Dict[str, Any]:
        """Send a test message ping to Slack or Discord webhook."""
        if not url or not url.strip().startswith("http"):
            return {"success": False, "error": "Invalid or empty Webhook URL"}

        payload = json.dumps({"text": "🚀 *LLM Fine-Tuning Studio*: Webhook alert connection verified successfully!"}).encode("utf-8")
        req = urllib.request.Request(
            url.strip(),
            data=payload,
            headers={"Content-Type": "application/json", "User-Agent": "FineTuningStudio/1.0"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as response:
                return {
                    "success": response.status in [200, 204],
                    "status_code": response.status,
                    "message": "Webhook ping sent successfully!",
                }
        except urllib.error.HTTPError as e:
            return {"success": False, "error": f"Webhook returned HTTP {e.code}"}
        except Exception as e:
            return {"success": False, "error": f"Webhook connection failed: {str(e)}"}


# Singleton instance
config_service = ConfigService()
