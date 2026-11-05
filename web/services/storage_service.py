"""
web/services/storage_service.py
-------------------------------
Unified object storage client supporting MinIO / AWS S3 with transparent
fallback to local filesystem storage for non-Dockerized execution.
"""

import os
import shutil
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from pipeline.common import PROJECT_ROOT

logger = logging.getLogger("doctune.storage")

HAS_BOTO3 = False
try:
    import boto3
    from botocore.client import Config as BotoConfig
    HAS_BOTO3 = True
except ImportError:
    pass


class StorageService:
    def __init__(self):
        self.endpoint_url = os.getenv("S3_ENDPOINT_URL")
        self.access_key = os.getenv("S3_ACCESS_KEY") or os.getenv("MINIO_ROOT_USER")
        self.secret_key = os.getenv("S3_SECRET_KEY") or os.getenv("MINIO_ROOT_PASSWORD")
        self.region = os.getenv("S3_REGION", "us-east-1")

        self.bucket_docs = os.getenv("S3_BUCKET_DOCS", "doc-tune-raw-docs")
        self.bucket_datasets = os.getenv("S3_BUCKET_DATASETS", "doc-tune-datasets")
        self.bucket_checkpoints = os.getenv("S3_BUCKET_CHECKPOINTS", "doc-tune-checkpoints")
        self.bucket_merged = os.getenv("S3_BUCKET_MERGED", "doc-tune-merged-models")

        self.s3_client = None
        if HAS_BOTO3 and self.endpoint_url and self.access_key and self.secret_key:
            try:
                self.s3_client = boto3.client(
                    "s3",
                    endpoint_url=self.endpoint_url,
                    aws_access_key_id=self.access_key,
                    aws_secret_access_key=self.secret_key,
                    region_name=self.region,
                    config=BotoConfig(signature_version="s3v4"),
                )
                logger.info("MinIO/S3 storage client connected: %s", self.endpoint_url)
            except Exception as e:
                logger.warning("Failed to initialize MinIO client, falling back to local disk: %s", e)
                self.s3_client = None

    def is_s3_enabled(self) -> bool:
        """Check if MinIO/S3 object storage is active."""
        return self.s3_client is not None

    def ensure_buckets_exist(self) -> List[str]:
        """Ensure all required DocTune buckets exist in S3/MinIO or local filesystem."""
        required_buckets = [
            self.bucket_docs,
            self.bucket_datasets,
            self.bucket_checkpoints,
            self.bucket_merged,
        ]
        ensured = []
        if self.is_s3_enabled():
            try:
                resp = self.s3_client.list_buckets()
                existing = [b["Name"] for b in resp.get("Buckets", [])]
                for bucket in required_buckets:
                    if bucket not in existing:
                        try:
                            self.s3_client.create_bucket(Bucket=bucket)
                            logger.info("Created MinIO S3 bucket: %s", bucket)
                        except Exception as e:
                            logger.warning("Bucket %s already exists or conflict: %s", bucket, e)
                    ensured.append(bucket)
            except Exception as e:
                logger.error("Failed to list/create S3 buckets: %s", e)
        else:
            for bucket in required_buckets:
                target_dir = PROJECT_ROOT / ".storage" / bucket
                target_dir.mkdir(parents=True, exist_ok=True)
                ensured.append(bucket)
        return ensured

    def upload_file(self, local_path: Path, bucket: str, s3_key: str) -> bool:
        """Upload a local file to MinIO/S3, or copy to local folder fallback."""
        if not local_path.exists():
            return False

        if self.is_s3_enabled():
            try:
                self.s3_client.upload_file(str(local_path), bucket, s3_key)
                return True
            except Exception as e:
                logger.error("S3 upload failed for %s/%s: %s", bucket, s3_key, e)
                return False
        else:
            # Local filesystem fallback
            target_dir = PROJECT_ROOT / ".storage" / bucket
            target_dir.mkdir(parents=True, exist_ok=True)
            target_path = target_dir / s3_key
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local_path, target_path)
            return True

    def download_file(self, bucket: str, s3_key: str, local_path: Path) -> bool:
        """Download a file from MinIO/S3, or retrieve from local storage fallback."""
        local_path.parent.mkdir(parents=True, exist_ok=True)

        if self.is_s3_enabled():
            try:
                self.s3_client.download_file(bucket, s3_key, str(local_path))
                return True
            except Exception as e:
                logger.error("S3 download failed for %s/%s: %s", bucket, s3_key, e)
                return False
        else:
            # Local filesystem fallback
            source_path = PROJECT_ROOT / ".storage" / bucket / s3_key
            if source_path.exists():
                shutil.copy2(source_path, local_path)
                return True
            return False

    def generate_presigned_download_url(self, bucket: str, s3_key: str, expires_in: int = 3600) -> Optional[str]:
        """Generate a presigned S3 download URL for direct browser downloading."""
        if not self.is_s3_enabled():
            return None
        try:
            url = self.s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket, "Key": s3_key},
                ExpiresIn=expires_in,
            )
            return url
        except Exception as e:
            logger.error("Failed to generate presigned URL: %s", e)
            return None

    def list_objects(self, bucket: str, prefix: str = "") -> List[Dict[str, Any]]:
        """List objects in an S3 bucket or local folder fallback."""
        results = []
        if self.is_s3_enabled():
            try:
                paginator = self.s3_client.get_paginator("list_objects_v2")
                for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                    for item in page.get("Contents", []):
                        results.append({
                            "key": item["Key"],
                            "size": item["Size"],
                            "last_modified": item["LastModified"].isoformat(),
                        })
            except Exception as e:
                logger.error("Failed to list S3 objects in %s: %s", bucket, e)
        else:
            source_dir = PROJECT_ROOT / ".storage" / bucket
            if source_dir.exists():
                prefix_clean = prefix.strip("/")
                for p in source_dir.rglob("*"):
                    if p.is_file():
                        rel = p.relative_to(source_dir).as_posix()
                        if not prefix_clean or rel.startswith(prefix_clean):
                            results.append({
                                "key": rel,
                                "size": p.stat().st_size,
                                "last_modified": None,
                            })
        return results

    def delete_file(self, bucket: str, s3_key: str) -> bool:
        """Delete an object from MinIO/S3 or local fallback storage."""
        if self.is_s3_enabled():
            try:
                self.s3_client.delete_object(Bucket=bucket, Key=s3_key)
                return True
            except Exception as e:
                logger.error("Failed to delete S3 object %s/%s: %s", bucket, s3_key, e)
                return False
        else:
            target_path = PROJECT_ROOT / ".storage" / bucket / s3_key
            if target_path.exists():
                try:
                    target_path.unlink()
                    return True
                except Exception as e:
                    logger.error("Failed to delete local fallback file %s: %s", target_path, e)
                    return False
            return False

    def delete_dir(self, bucket: str, prefix: str) -> bool:
        """Delete an entire prefix folder from MinIO/S3 or local fallback."""
        if not prefix:
            return False
        if self.is_s3_enabled():
            try:
                objs = self.list_objects(bucket, prefix=prefix)
                if objs:
                    delete_keys = [{"Key": o["key"]} for o in objs]
                    for i in range(0, len(delete_keys), 1000):
                        batch = delete_keys[i:i + 1000]
                        self.s3_client.delete_objects(Bucket=bucket, Delete={"Objects": batch})
                return True
            except Exception as e:
                logger.error("Failed to delete S3 prefix %s/%s: %s", bucket, prefix, e)
                return False
        else:
            target_dir = PROJECT_ROOT / ".storage" / bucket / prefix
            if target_dir.exists():
                try:
                    shutil.rmtree(target_dir)
                    return True
                except Exception as e:
                    logger.error("Failed to delete local fallback dir %s: %s", target_dir, e)
                    return False
            return False

    def upload_dir(self, local_dir: Path, bucket: str, s3_prefix: str = "") -> bool:
        """Recursively upload all files in a directory to MinIO/S3 preserving tree structure."""
        if not local_dir.exists() or not local_dir.is_dir():
            return False
        files = [p for p in local_dir.rglob("*") if p.is_file()]
        if not files:
            return False

        success = True
        prefix_clean = s3_prefix.strip("/")
        for f in files:
            rel = f.relative_to(local_dir).as_posix()
            s3_key = f"{prefix_clean}/{rel}" if prefix_clean else rel
            if not self.upload_file(f, bucket, s3_key):
                success = False
        return success

    def download_dir(self, bucket: str, s3_prefix: str, local_dir: Path) -> bool:
        """Download all files under an S3 prefix to local directory preserving structure."""
        local_dir.mkdir(parents=True, exist_ok=True)
        objs = self.list_objects(bucket, prefix=s3_prefix)
        if not objs:
            return False

        prefix_clean = s3_prefix.strip("/")
        prefix_prefix = f"{prefix_clean}/" if prefix_clean else ""

        success = True
        for o in objs:
            key = o["key"]
            rel = key[len(prefix_prefix):] if prefix_prefix and key.startswith(prefix_prefix) else Path(key).name
            target_path = local_dir / rel
            if not self.download_file(bucket, key, target_path):
                success = False
        return success

    def sync_bucket_to_dir(self, bucket: str, local_dir: Path, prefix: str = "") -> int:
        """Synchronize all objects in a bucket prefix down to a local directory."""
        local_dir.mkdir(parents=True, exist_ok=True)
        objs = self.list_objects(bucket, prefix=prefix)
        count = 0
        prefix_clean = prefix.strip("/")
        prefix_prefix = f"{prefix_clean}/" if prefix_clean else ""

        for o in objs:
            key = o["key"]
            rel = key[len(prefix_prefix):] if prefix_prefix and key.startswith(prefix_prefix) else Path(key).name
            target_path = local_dir / rel
            if self.download_file(bucket, key, target_path):
                count += 1
        return count


storage_service = StorageService()
