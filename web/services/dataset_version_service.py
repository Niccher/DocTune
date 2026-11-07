"""
web/services/dataset_version_service.py
---------------------------------------
Dataset snapshotting, immutable versioning, and provenance tracking.
Enables tracing any fine-tuned model checkpoint back to the exact training dataset.
"""

import os
import json
import time
import shutil
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

from pipeline.common import PROJECT_ROOT, DATA_DIR, DEFAULT_JSONL

logger = logging.getLogger("doctune.dataset_version")

DATASETS_DIR = DATA_DIR / "datasets"
MANIFEST_FILE = DATASETS_DIR / "manifest.json"


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class DatasetVersionService:
    def __init__(self):
        DATASETS_DIR.mkdir(parents=True, exist_ok=True)
        self._ensure_manifest()

    def _ensure_manifest(self):
        if not MANIFEST_FILE.exists():
            with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
                json.dump({"active_version": None, "versions": []}, f, indent=2)

    def _read_manifest(self) -> Dict[str, Any]:
        self._ensure_manifest()
        try:
            with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error("Failed to read dataset manifest: %s", e)
            return {"active_version": None, "versions": []}

    def _write_manifest(self, data: Dict[str, Any]):
        DATASETS_DIR.mkdir(parents=True, exist_ok=True)
        with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def compute_sha256(self, file_path: Path) -> str:
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def create_version_from_file(
        self,
        source_file: Path,
        tag: Optional[str] = None,
        description: Optional[str] = None,
        set_active: bool = True,
    ) -> Dict[str, Any]:
        """Create an immutable, versioned copy of a JSONL dataset file."""
        if not source_file.exists():
            raise FileNotFoundError(f"Source dataset not found: {source_file}")

        file_hash = self.compute_sha256(source_file)
        file_size = source_file.stat().st_size
        ts = int(time.time())
        iso_now = now_utc_iso()

        manifest = self._read_manifest()
        version_num = len(manifest.get("versions", [])) + 1
        version_tag = tag.strip() if tag and tag.strip() else f"v{version_num}"
        version_id = f"ds_{version_tag}_{ts}_{file_hash[:8]}"

        # Analyze dataset samples
        sample_count = 0
        total_tokens_est = 0
        try:
            with open(source_file, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    sample_count += 1
                    try:
                        obj = json.loads(line)
                        for m in obj.get("messages", []):
                            content = m.get("content", "")
                            total_tokens_est += len(content) // 4
                    except Exception:
                        pass
        except Exception as e:
            logger.warning("Error inspecting dataset samples: %s", e)

        dest_filename = f"{version_id}.jsonl"
        dest_path = DATASETS_DIR / dest_filename
        shutil.copy2(source_file, dest_path)

        entry = {
            "version_id": version_id,
            "version_tag": version_tag,
            "filename": dest_filename,
            "path": str(dest_path),
            "sha256": file_hash,
            "size_bytes": file_size,
            "sample_count": sample_count,
            "est_tokens": total_tokens_est,
            "description": description or f"Snapshot {version_tag} with {sample_count} samples",
            "created_at": iso_now,
            "is_active": set_active,
        }

        # Update manifest
        versions = manifest.get("versions", [])
        if set_active:
            for v in versions:
                v["is_active"] = False
            manifest["active_version"] = version_id

        versions.insert(0, entry)
        manifest["versions"] = versions
        self._write_manifest(manifest)

        # Update symlink or default file
        if set_active and source_file != DEFAULT_JSONL:
            try:
                shutil.copy2(dest_path, DEFAULT_JSONL)
            except Exception as e:
                logger.warning("Failed to update DEFAULT_JSONL alias: %s", e)

        # Record in DB if available
        try:
            from web.db.session import SessionLocal
            from web.db.models import DatasetVersion
            if SessionLocal:
                db = SessionLocal()
                try:
                    db_entry = DatasetVersion(
                        id=version_id,
                        version_tag=version_tag,
                        file_path=str(dest_path),
                        file_hash=file_hash,
                        size_bytes=file_size,
                        sample_count=sample_count,
                        description=description,
                    )
                    db.add(db_entry)
                    db.commit()
                except Exception as ex:
                    logger.debug("DatasetVersion DB insert skipped: %s", ex)
                finally:
                    db.close()
        except Exception:
            pass

        return entry

    def list_versions(self) -> List[Dict[str, Any]]:
        manifest = self._read_manifest()
        return manifest.get("versions", [])

    def get_version(self, version_id: str) -> Optional[Dict[str, Any]]:
        for v in self.list_versions():
            if v["version_id"] == version_id or v["version_tag"] == version_id:
                return v
        return None

    def activate_version(self, version_id: str) -> bool:
        manifest = self._read_manifest()
        target = None
        for v in manifest.get("versions", []):
            if v["version_id"] == version_id or v["version_tag"] == version_id:
                target = v
                v["is_active"] = True
            else:
                v["is_active"] = False

        if not target:
            return False

        manifest["active_version"] = target["version_id"]
        self._write_manifest(manifest)

        target_path = Path(target["path"])
        if target_path.exists():
            try:
                shutil.copy2(target_path, DEFAULT_JSONL)
            except Exception as e:
                logger.error("Failed to copy activated dataset to DEFAULT_JSONL: %s", e)
                return False
        return True

    def delete_version(self, version_id: str) -> bool:
        manifest = self._read_manifest()
        versions = manifest.get("versions", [])
        new_versions = [v for v in versions if v["version_id"] != version_id and v["version_tag"] != version_id]
        if len(new_versions) == len(versions):
            return False
        if manifest.get("active_version") == version_id:
            manifest["active_version"] = new_versions[0]["version_id"] if new_versions else None
        manifest["versions"] = new_versions
        self._write_manifest(manifest)
        return True

    def get_active_version(self) -> Optional[Dict[str, Any]]:
        manifest = self._read_manifest()
        active_id = manifest.get("active_version")
        if active_id:
            return self.get_version(active_id)
        versions = manifest.get("versions", [])
        return versions[0] if versions else None


dataset_version_service = DatasetVersionService()
