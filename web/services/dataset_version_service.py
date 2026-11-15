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
        metadata: Optional[Dict[str, Any]] = None,
        custom_filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create an immutable, versioned copy of a JSONL dataset file and upload to MinIO."""
        if not source_file.exists():
            raise FileNotFoundError(f"Source dataset not found: {source_file}")

        file_hash = self.compute_sha256(source_file)
        file_size = source_file.stat().st_size
        ts = int(time.time())
        iso_now = now_utc_iso()
        meta = metadata or {}

        manifest = self._read_manifest()
        version_num = len(manifest.get("versions", [])) + 1
        version_tag = tag.strip() if tag and tag.strip() else f"v{version_num}"
        version_id = f"ds_{version_tag}_{ts}_{file_hash[:8]}"

        # Analyze dataset samples
        sample_count = 0
        total_tokens_est = 0
        topic_preview = None
        try:
            with open(source_file, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    sample_count += 1
                    try:
                        obj = json.loads(line)
                        msgs = obj.get("messages", [])
                        for m in msgs:
                            content = m.get("content", "")
                            total_tokens_est += len(content) // 4
                            if topic_preview is None and m.get("role") == "user":
                                topic_preview = content[:60]
                    except Exception:
                        pass
        except Exception as e:
            logger.warning("Error inspecting dataset samples: %s", e)

        dest_filename = custom_filename or f"{version_id}.jsonl"
        dest_path = DATASETS_DIR / dest_filename
        shutil.copy2(source_file, dest_path)

        # Upload to MinIO / S3
        from web.services.storage_service import storage_service
        s3_key = f"datasets/{dest_filename}"
        s3_tag_key = f"datasets/{version_tag}.jsonl"
        storage_service.upload_file(dest_path, storage_service.bucket_datasets, s3_key)
        if s3_key != s3_tag_key:
            storage_service.upload_file(dest_path, storage_service.bucket_datasets, s3_tag_key)

        entry = {
            "version_id": version_id,
            "version_tag": version_tag,
            "filename": dest_filename,
            "path": str(dest_path),
            "sha256": file_hash,
            "size_bytes": file_size,
            "size_kb": round(file_size / 1024, 1),
            "sample_count": sample_count,
            "est_tokens": total_tokens_est,
            "description": description or f"Snapshot {version_tag} with {sample_count} samples",
            "created_at": iso_now,
            "is_active": set_active,
            "teacher_model": meta.get("teacher_model", "Unknown"),
            "teacher_provider": meta.get("teacher_provider", "Universal"),
            "duration_sec": meta.get("duration_sec"),
            "s3_bucket": storage_service.bucket_datasets,
            "s3_key": s3_key,
            "topic_preview": topic_preview or "Enterprise Q&A",
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
                        version_tag=version_tag[:32],
                        file_path=str(dest_path),
                        file_hash=file_hash,
                        size_bytes=file_size,
                        sample_count=sample_count,
                        description=description,
                        s3_key=s3_key,
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

    def create_auto_version(
        self,
        source_file: Path,
        slug_hint: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        set_active: bool = True,
    ) -> Dict[str, Any]:
        """
        Auto-save dataset using format: YYYY-Mon-DD-HHMMSS_<slug>.jsonl
        and automatically register to MinIO and version registry.
        """
        import re
        now = datetime.now()
        timestamp_str = now.strftime("%Y-%b-%d-%H%M%S")

        raw_slug = (slug_hint or "dataset").strip().lower()
        clean_slug = re.sub(r"[^a-zA-Z0-9_-]", "_", raw_slug)
        clean_slug = re.sub(r"_+", "_", clean_slug).strip("_")[:32] or "qa_data"

        # Cap version_tag to 32 chars to strictly comply with database schema Column(String(32))
        auto_tag = f"{timestamp_str}_{clean_slug}"[:32]
        custom_filename = f"{timestamp_str}_{clean_slug}.jsonl"

        desc = description or f"Auto-saved dataset snapshot on {now.strftime('%Y-%m-%d %H:%M:%S')}"
        return self.create_version_from_file(
            source_file=source_file,
            tag=auto_tag,
            description=desc,
            set_active=set_active,
            metadata=metadata,
            custom_filename=custom_filename,
        )

    def list_versions(self) -> List[Dict[str, Any]]:
        manifest = self._read_manifest()
        return manifest.get("versions", [])

    def get_version(self, version_id: str) -> Optional[Dict[str, Any]]:
        for v in self.list_versions():
            if v["version_id"] == version_id or v["version_tag"] == version_id or v.get("filename") == version_id:
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
        target = None
        new_versions = []
        for v in versions:
            if v["version_id"] == version_id or v["version_tag"] == version_id or v.get("filename") == version_id:
                target = v
            else:
                new_versions.append(v)

        if not target:
            return False

        if manifest.get("active_version") in (version_id, target.get("version_id")):
            if new_versions:
                manifest["active_version"] = new_versions[0]["version_id"]
                new_versions[0]["is_active"] = True
                new_path = Path(new_versions[0].get("path", ""))
                if new_path.exists():
                    try:
                        shutil.copy2(new_path, DEFAULT_JSONL)
                    except Exception as e:
                        logger.warning("Failed to update DEFAULT_JSONL on snapshot delete: %s", e)
            else:
                manifest["active_version"] = None

        manifest["versions"] = new_versions
        self._write_manifest(manifest)

        # Delete from MinIO S3
        from web.services.storage_service import storage_service
        s3_key = target.get("s3_key") or f"datasets/{target.get('filename')}"
        storage_service.delete_file(storage_service.bucket_datasets, s3_key)
        if target.get("version_tag"):
            storage_service.delete_file(storage_service.bucket_datasets, f"datasets/{target.get('version_tag')}.jsonl")

        # Delete local file
        target_path = Path(target.get("path", ""))
        if target_path.exists():
            try:
                target_path.unlink()
            except Exception as e:
                logger.warning("Could not delete local snapshot file: %s", e)

        # Delete from DB
        try:
            from web.db.session import SessionLocal
            from web.db.models import DatasetVersion
            if SessionLocal:
                db = SessionLocal()
                try:
                    db.query(DatasetVersion).filter(
                        (DatasetVersion.id == version_id) | (DatasetVersion.version_tag == version_id)
                    ).delete()
                    db.commit()
                except Exception as ex:
                    logger.debug("DatasetVersion DB delete skipped: %s", ex)
                finally:
                    db.close()
        except Exception:
            pass

        return True

    def get_active_version(self) -> Optional[Dict[str, Any]]:
        manifest = self._read_manifest()
        active_id = manifest.get("active_version")
        if active_id:
            return self.get_version(active_id)
        versions = manifest.get("versions", [])
        return versions[0] if versions else None


dataset_version_service = DatasetVersionService()
