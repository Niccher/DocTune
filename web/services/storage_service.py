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
                resp = self.s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
                for item in resp.get("Contents", []):
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
                for p in source_dir.rglob("*"):
                    if p.is_file():
                        rel = p.relative_to(source_dir)
                        results.append({
                            "key": str(rel),
                            "size": p.stat().st_size,
                            "last_modified": None,
                        })
        return results


storage_service = StorageService()
