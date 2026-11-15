"""
web/routers/rag.py
------------------
REST API endpoints for RAG vector index lifecycle management,
direct semantic search testing, and retrieval diagnostics.
"""

from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Request, Form, Query, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from web.services.rag_service import rag_service
from web.routers.auth import get_project_id, require_user
from web.celery_app import is_celery_available, app as celery_app

router = APIRouter(prefix="/api/v1/rag", tags=["RAG Subsystem"])


class RagQueryRequest(BaseModel):
    query: str = Field(..., description="Query prompt to retrieve matching chunks for")
    top_k: Optional[int] = Field(default=3, ge=1, le=20)
    min_score: Optional[float] = Field(default=0.25, ge=0.0, le=1.0)


@router.get("/status", response_class=JSONResponse)
async def get_rag_status(request: Request):
    """Retrieve vector store chunk counts, persistence location, and readiness status."""
    project_id = get_project_id(request)
    stats = rag_service.get_index_status(project_id=project_id)
    return JSONResponse({"status": "success", "data": stats})


@router.get("/chunks", response_class=JSONResponse)
async def get_rag_chunks_api(request: Request, limit: int = 150, offset: int = 0):
    """Retrieve full text of indexed chunks stored in ChromaDB for user inspection."""
    project_id = get_project_id(request)
    chunks = rag_service.get_all_chunks(project_id=project_id, limit=limit, offset=offset)
    stats = rag_service.get_index_status(project_id=project_id)
    return JSONResponse({
        "status": "success",
        "total_chunks": stats.get("total_chunks", 0),
        "count": len(chunks),
        "chunks": chunks,
        "data": {
            "total_chunks": stats.get("total_chunks", 0),
            "count": len(chunks),
            "chunks": chunks,
        },
    })


@router.post("/index/build", response_class=JSONResponse)
async def build_rag_index(request: Request, _user=Depends(require_user)):
    """
    Trigger full vector indexing of all uploaded enterprise documents.
    Dispatches to Celery data_prep_queue if distributed cluster is available,
    otherwise executes synchronously.
    """
    project_id = get_project_id(request)

    # If Celery worker is available, dispatch as async background task
    if is_celery_available() and celery_app:
        task = celery_app.send_task(
            "web.celery_app.task_build_rag_index",
            args=[project_id],
            queue="data_prep_queue",
        )
        return JSONResponse({
            "status": "started",
            "mode": "celery",
            "task_id": task.id,
            "message": "RAG vector indexing dispatched to background worker queue.",
        })

    # Synchronous execution fallback
    try:
        result = rag_service.build_index(project_id=project_id)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@router.post("/query", response_class=JSONResponse)
async def query_rag_chunks(payload: RagQueryRequest, request: Request):
    """Direct semantic retrieval endpoint returning top-K chunks and citations."""
    project_id = get_project_id(request)
    if not payload.query or not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    hits = rag_service.search(
        query=payload.query,
        project_id=project_id,
        top_k=payload.top_k or 3,
        min_score=payload.min_score or 0.25,
    )
    context_str, citations = rag_service.retriever.assemble_context(hits)

    return JSONResponse({
        "status": "success",
        "query": payload.query,
        "total_hits": len(hits),
        "hits": hits,
        "citations": citations,
        "assembled_context": context_str,
    })


@router.delete("/index", response_class=JSONResponse)
async def clear_rag_index(request: Request):
    """Clear all vectorized chunks and reset the project's vector store."""
    project_id = get_project_id(request)
    cleared = rag_service.clear_index(project_id=project_id)
    return JSONResponse({"status": "success" if cleared else "not_found", "project_id": project_id})
