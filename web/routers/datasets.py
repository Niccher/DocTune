"""
web/routers/datasets.py
-----------------------
Universal dataset management, multi-format ingestion, token distribution analysis,
truncation risk calculation, and immutable snapshot versioning.
"""

import os
import time
import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, Request, UploadFile, File, Form, Query, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel

from pipeline.common import PROJECT_ROOT, DATA_DIR
from web.core.ids import generate_ulid
from web.db.session import get_db
from web.db.models import Dataset, DatasetVersion, User
from web.services.data.importer import dataset_importer
from web.services.data.validator import dataset_validator
from web.services.storage_service import storage_service
from web.services.audit_service import audit_service
from web.routers.auth import require_project_role

logger = logging.getLogger("doctune.datasets")
router = APIRouter(prefix="/api/v1/datasets", tags=["datasets"])

DATASETS_STORAGE_DIR = DATA_DIR / "datasets"
DATASETS_STORAGE_DIR.mkdir(parents=True, exist_ok=True)


def get_project_id(request: Request) -> str:
    """Helper to extract active project ID from request state or fallback."""
    return getattr(request.state, "project_id", "proj_default")


def get_user_id(request: Request) -> Optional[int]:
    user = getattr(request.state, "user", None)
    return user.id if user else None


def compute_file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


class CreateDatasetRequest(BaseModel):
    name: str
    description: Optional[str] = None
    source_format: str = "jsonl"


class CurateDatasetRequest(BaseModel):
    source_version_id: str
    target_tag: str
    description: Optional[str] = None
    deduplicate: bool = True
    max_tokens: Optional[int] = 2048
    min_tokens: Optional[int] = 10


@router.get("", response_class=JSONResponse)
def list_datasets(
    request: Request,
    db: Session = Depends(get_db),
    _role=Depends(require_project_role("viewer")),
):
    """List all datasets in the current project context with version count and latest stats."""
    project_id = get_project_id(request)
    datasets = (
        db.query(Dataset)
        .filter(Dataset.project_id == project_id)
        .order_by(Dataset.created_at.desc())
        .all()
    )

    results = []
    for d in datasets:
        versions = (
            db.query(DatasetVersion)
            .filter(DatasetVersion.dataset_id == d.id)
            .order_by(DatasetVersion.created_at.desc())
            .all()
        )
        latest_version = versions[0] if versions else None
        results.append({
            "id": d.id,
            "project_id": d.project_id,
            "name": d.name,
            "description": d.description,
            "source_format": d.source_format,
            "created_at": d.created_at.isoformat() if d.created_at else None,
            "version_count": len(versions),
            "latest_version": {
                "id": latest_version.id,
                "version_tag": latest_version.version_tag,
                "sample_count": latest_version.sample_count,
                "size_bytes": latest_version.size_bytes,
                "file_hash": latest_version.file_hash,
                "created_at": latest_version.created_at.isoformat() if latest_version.created_at else None,
                "stats": latest_version.stats_json,
            } if latest_version else None,
        })

    return JSONResponse({"status": "success", "datasets": results})


@router.post("", response_class=JSONResponse)
def create_dataset(
    req: CreateDatasetRequest,
    request: Request,
    db: Session = Depends(get_db),
    _role=Depends(require_project_role("operator")),
):
    """Create a new logical dataset container."""
    project_id = get_project_id(request)
    user_id = get_user_id(request)
    dataset_id = generate_ulid("ds_")

    ds = Dataset(
        id=dataset_id,
        project_id=project_id,
        name=req.name.strip(),
        description=req.description.strip() if req.description else None,
        source_format=req.source_format.strip().lower(),
        created_by=user_id,
    )
    db.add(ds)
    db.commit()
    db.refresh(ds)

    audit_service.log_action("CREATE_DATASET", actor_name=f"User:{user_id or 'anon'}", details=f"Dataset {ds.name} ({ds.id})")
    return JSONResponse({"status": "success", "dataset_id": ds.id, "name": ds.name})


@router.post("/import", response_class=JSONResponse)
async def import_dataset(
    request: Request,
    file: UploadFile = File(...),
    dataset_name: Optional[str] = Form(None),
    dataset_id: Optional[str] = Form(None),
    version_tag: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    column_mapping: Optional[str] = Form(None),
    db: Session = Depends(get_db),
    _role=Depends(require_project_role("operator")),
):
    """
    Import a file (JSONL, CSV, etc.), normalize to canonical OpenAI chat format,
    compute token distributions & truncation risk, and store as an immutable DatasetVersion.
    """
    project_id = get_project_id(request)
    user_id = get_user_id(request)

    col_map = None
    if column_mapping:
        try:
            col_map = json.loads(column_mapping)
        except Exception:
            pass

    # Save uploaded file to temp path
    upload_temp_dir = DATASETS_STORAGE_DIR / "uploads"
    upload_temp_dir.mkdir(parents=True, exist_ok=True)
    temp_raw_path = upload_temp_dir / f"raw_{int(time.time())}_{file.filename}"

    with open(temp_raw_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)

    # 1. Resolve or create Dataset
    if dataset_id:
        ds = db.query(Dataset).filter(Dataset.id == dataset_id, Dataset.project_id == project_id).first()
        if not ds:
            raise HTTPException(status_code=404, detail="Dataset container not found in current project")
    else:
        name = dataset_name.strip() if dataset_name and dataset_name.strip() else Path(file.filename).stem
        dataset_id = generate_ulid("ds_")
        ds = Dataset(
            id=dataset_id,
            project_id=project_id,
            name=name,
            description=description,
            source_format=Path(file.filename).suffix.lstrip(".").lower() or "jsonl",
            created_by=user_id,
        )
        db.add(ds)
        db.commit()
        db.refresh(ds)

    # 2. Normalize via DatasetImporter
    tag = version_tag.strip() if version_tag and version_tag.strip() else f"v{int(time.time())}"
    normalized_filename = f"{ds.id}_{tag}_canonical.jsonl"
    normalized_path = DATASETS_STORAGE_DIR / normalized_filename

    row_count, detected_schema = dataset_importer.import_file(
        input_path=temp_raw_path,
        output_path=normalized_path,
        column_mapping=col_map,
    )

    if row_count == 0:
        if normalized_path.exists():
            normalized_path.unlink()
        raise HTTPException(
            status_code=400,
            detail="Failed to extract any valid training records from the uploaded file.",
        )

    # 3. Comprehensive Validation & Token Distribution Analysis
    val_report = dataset_validator.validate_file(normalized_path)
    file_hash = compute_file_sha256(normalized_path)
    file_size = normalized_path.stat().st_size
    version_id = f"ver_{generate_ulid()}"

    # 4. MinIO / S3 backup if enabled
    s3_key = None
    if storage_service.is_s3_enabled():
        s3_key = f"projects/{project_id}/datasets/{ds.id}/{version_id}.jsonl"
        storage_service.upload_file(normalized_path, storage_service.bucket_datasets, s3_key)

    # 5. Persist DatasetVersion
    ver = DatasetVersion(
        id=version_id,
        dataset_id=ds.id,
        project_id=project_id,
        created_by=user_id,
        version_tag=tag,
        file_path=str(normalized_path),
        s3_key=s3_key,
        file_hash=file_hash,
        split_type="train",
        size_bytes=file_size,
        sample_count=val_report.get("valid_samples", row_count),
        stats_json=val_report,
        description=description,
    )
    db.add(ver)
    db.commit()
    db.refresh(ver)

    audit_service.log_action(
        "IMPORT_DATASET",
        actor_name=f"User:{user_id or 'anon'}",
        details=f"Imported {row_count} samples for {ds.name} [{ver.version_tag}] (hash: {file_hash[:8]})",
    )

    # Clean up raw upload
    try:
        temp_raw_path.unlink(missing_ok=True)
    except Exception:
        pass

    return JSONResponse({
        "status": "success",
        "dataset_id": ds.id,
        "version_id": ver.id,
        "version_tag": ver.version_tag,
        "sample_count": ver.sample_count,
        "file_hash": file_hash,
        "detected_schema": detected_schema,
        "validation": val_report,
    })


@router.post("/validate", response_class=JSONResponse)
async def validate_dataset_upload(
    request: Request,
    file: UploadFile = File(...),
    column_mapping: Optional[str] = Form(None),
    _role=Depends(require_project_role("viewer")),
):
    """Validate and analyze a dataset file without saving it to database."""
    col_map = None
    if column_mapping:
        try:
            col_map = json.loads(column_mapping)
        except Exception:
            pass

    temp_dir = DATASETS_STORAGE_DIR / "temp_val"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_raw = temp_dir / f"test_{int(time.time())}_{file.filename}"
    temp_norm = temp_dir / f"test_norm_{int(time.time())}.jsonl"

    try:
        with open(temp_raw, "wb") as buf:
            buf.write(await file.read())

        row_count, schema = dataset_importer.import_file(temp_raw, temp_norm, column_mapping=col_map, max_rows=1000)
        report = dataset_validator.validate_file(temp_norm)
        report["detected_schema"] = schema
        report["sampled_rows"] = row_count
        return JSONResponse({"status": "success", "report": report})
    finally:
        temp_raw.unlink(missing_ok=True)
        temp_norm.unlink(missing_ok=True)


@router.get("/{dataset_id}", response_class=JSONResponse)
def get_dataset(
    dataset_id: str,
    request: Request,
    db: Session = Depends(get_db),
    _role=Depends(require_project_role("viewer")),
):
    """Get dataset metadata and all immutable versions."""
    project_id = get_project_id(request)
    ds = db.query(Dataset).filter(Dataset.id == dataset_id, Dataset.project_id == project_id).first()
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")

    versions = (
        db.query(DatasetVersion)
        .filter(DatasetVersion.dataset_id == ds.id)
        .order_by(DatasetVersion.created_at.desc())
        .all()
    )

    return JSONResponse({
        "status": "success",
        "dataset": {
            "id": ds.id,
            "project_id": ds.project_id,
            "name": ds.name,
            "description": ds.description,
            "source_format": ds.source_format,
            "created_at": ds.created_at.isoformat() if ds.created_at else None,
            "versions": [
                {
                    "id": v.id,
                    "version_tag": v.version_tag,
                    "sample_count": v.sample_count,
                    "size_bytes": v.size_bytes,
                    "file_hash": v.file_hash,
                    "split_type": v.split_type,
                    "stats": v.stats_json,
                    "created_at": v.created_at.isoformat() if v.created_at else None,
                }
                for v in versions
            ],
        },
    })


@router.get("/{dataset_id}/versions/{version_id}/samples", response_class=JSONResponse)
def preview_version_samples(
    dataset_id: str,
    version_id: str,
    offset: int = Query(0, ge=0),
    limit: int = Query(25, ge=1, le=100),
    request: Request = None,
    db: Session = Depends(get_db),
    _role=Depends(require_project_role("viewer")),
):
    """Preview records from an immutable dataset snapshot with token distributions."""
    project_id = get_project_id(request)
    ver = (
        db.query(DatasetVersion)
        .filter(DatasetVersion.id == version_id, DatasetVersion.dataset_id == dataset_id, DatasetVersion.project_id == project_id)
        .first()
    )
    if not ver:
        raise HTTPException(status_code=404, detail="Dataset version not found")

    path = Path(ver.file_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Dataset file missing on disk")

    samples = []
    total = 0
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            if total >= offset and len(samples) < limit:
                try:
                    data = json.loads(line)
                    samples.append({
                        "index": idx,
                        "data": data,
                        "tokens_est": sum(len(m.get("content", "")) // 4 for m in data.get("messages", []))
                        if "messages" in data
                        else 0,
                    })
                except Exception:
                    pass
            total += 1

    return JSONResponse({
        "status": "success",
        "dataset_id": dataset_id,
        "version_id": version_id,
        "total_records": total,
        "offset": offset,
        "limit": limit,
        "samples": samples,
    })


@router.post("/{dataset_id}/curate", response_class=JSONResponse)
def curate_dataset(
    dataset_id: str,
    req: CurateDatasetRequest,
    request: Request,
    db: Session = Depends(get_db),
    _role=Depends(require_project_role("operator")),
):
    """
    Curate and create a new immutable dataset version from an existing version.
    Supports deduplication, length filtering, and quality thresholding.
    """
    project_id = get_project_id(request)
    user_id = get_user_id(request)

    src_ver = (
        db.query(DatasetVersion)
        .filter(DatasetVersion.id == req.source_version_id, DatasetVersion.dataset_id == dataset_id, DatasetVersion.project_id == project_id)
        .first()
    )
    if not src_ver:
        raise HTTPException(status_code=404, detail="Source dataset version not found")

    src_path = Path(src_ver.file_path)
    if not src_path.exists():
        raise HTTPException(status_code=404, detail="Source dataset file missing on disk")

    new_version_id = f"ver_{generate_ulid()}"
    new_filename = f"{dataset_id}_{req.target_tag.strip()}_canonical.jsonl"
    new_path = DATASETS_STORAGE_DIR / new_filename

    seen_hashes = set()
    accepted = 0
    filtered_tokens = 0
    filtered_dupes = 0

    with open(src_path, "r", encoding="utf-8") as in_f, open(new_path, "w", encoding="utf-8") as out_f:
        for line in in_f:
            clean = line.strip()
            if not clean:
                continue

            if req.deduplicate:
                h = hashlib.sha256(clean.encode("utf-8")).hexdigest()
                if h in seen_hashes:
                    filtered_dupes += 1
                    continue
                seen_hashes.add(h)

            try:
                record = json.loads(clean)
                tokens = sum(len(m.get("content", "")) // 4 for m in record.get("messages", []))
                if req.max_tokens and tokens > req.max_tokens:
                    filtered_tokens += 1
                    continue
                if req.min_tokens and tokens < req.min_tokens:
                    filtered_tokens += 1
                    continue

                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                accepted += 1
            except Exception:
                pass

    if accepted == 0:
        new_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Curated filter produced 0 valid samples")

    val_report = dataset_validator.validate_file(new_path)
    new_hash = compute_file_sha256(new_path)
    new_size = new_path.stat().st_size

    new_ver = DatasetVersion(
        id=new_version_id,
        dataset_id=dataset_id,
        project_id=project_id,
        created_by=user_id,
        version_tag=req.target_tag.strip(),
        file_path=str(new_path),
        file_hash=new_hash,
        split_type="train",
        size_bytes=new_size,
        sample_count=accepted,
        stats_json=val_report,
        description=req.description or f"Curated from {src_ver.version_tag} (removed {filtered_dupes} dupes, {filtered_tokens} length outliers)",
    )
    db.add(new_ver)
    db.commit()
    db.refresh(new_ver)

    audit_service.log_action(
        "CURATE_DATASET",
        actor_name=f"User:{user_id or 'anon'}",
        details=f"Curated {accepted} samples for {dataset_id} [{new_ver.version_tag}]",
    )

    return JSONResponse({
        "status": "success",
        "new_version_id": new_ver.id,
        "version_tag": new_ver.version_tag,
        "sample_count": accepted,
        "filtered_duplicates": filtered_dupes,
        "filtered_tokens": filtered_tokens,
        "file_hash": new_hash,
        "validation": val_report,
    })
