"""
web/routers/inference.py
------------------------
OpenAI-compatible inference API endpoints (/v1/chat/completions, /v1/models).
Enforces API key authentication and routes requests to llama.cpp (default).
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Request, HTTPException, status, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from web.services.inference_service import inference_service
from web.services.inference.manager import inference_manager
from web.services.config_service import config_service
from web.routers.auth import require_project_role, get_project_id

router = APIRouter(tags=["Inference API"])


class ChatMessage(BaseModel):
    role: str
    content: str
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str = Field(default="default", description="ID of the model or checkpoint to use")
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    max_tokens: Optional[int] = 512
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0


def check_api_key_auth(request: Request):
    """Verify API key authentication if required."""
    auth_header = request.headers.get("Authorization")
    auth_cfg = config_service.get_auth_config()

    # If auth is explicitly enabled or an Authorization header is provided, validate it
    if auth_header:
        api_key_obj = inference_manager.authenticate_api_key(auth_header)
        if not api_key_obj:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired API key. Key format must be 'Bearer dt_live_...'",
            )
        request.state.api_key = api_key_obj
        request.state.project_id = api_key_obj.project_id
    elif auth_cfg.get("require_api_key_for_inference", False):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please provide 'Authorization: Bearer dt_live_...'",
        )


@router.post("/v1/chat/completions")
async def create_chat_completion(payload: ChatCompletionRequest, request: Request):
    """
    OpenAI-compatible chat completion endpoint.
    Supports both standard JSON responses and SSE streaming (stream=True).
    Backed by llama.cpp as the default inference engine.
    """
    check_api_key_auth(request)

    if not payload.messages:
        raise HTTPException(status_code=400, detail="Messages array cannot be empty")

    dict_messages = [{"role": m.role, "content": m.content} for m in payload.messages]

    if payload.stream:
        generator = inference_service.stream_completion(
            messages=dict_messages,
            model_name=payload.model,
            temperature=payload.temperature or 0.7,
            max_tokens=payload.max_tokens or 512,
        )
        return StreamingResponse(generator, media_type="text/event-stream")

    completion = await inference_service.generate_completion_async(
        messages=dict_messages,
        model_name=payload.model,
        temperature=payload.temperature or 0.7,
        max_tokens=payload.max_tokens or 512,
    )
    return JSONResponse(completion)


@router.get("/v1/models")
async def list_models(request: Request):
    """List available fine-tuned models, checkpoints, and base models in OpenAI schema."""
    models = inference_service.list_available_models()
    return JSONResponse({
        "object": "list",
        "data": models,
    })


@router.get("/v1/models/{model_id}")
async def retrieve_model(model_id: str, request: Request):
    """Retrieve details for a specific model ID."""
    models = inference_service.list_available_models()
    found = next((m for m in models if m["id"] == model_id), None)
    if not found:
        return JSONResponse({
            "id": model_id,
            "object": "model",
            "created": 1700000000,
            "owned_by": "doctune",
        })
    return JSONResponse(found)


@router.get("/api/v1/inference/engine-health")
async def get_engine_health(request: Request):
    """Expose live status of llama.cpp, Ollama, and PyTorch backends."""
    health = await inference_manager.get_system_health()
    return JSONResponse(health)


@router.post("/api/v1/inference/deploy", response_class=JSONResponse)
async def deploy_model_endpoint(request: Request, _role=Depends(require_project_role("operator"))):
    """Deploy a model or run checkpoint to active inference engine."""
    body = await request.json()
    run_id = body.get("run_id") or body.get("checkpoint_id")
    if not run_id:
        raise HTTPException(status_code=400, detail="run_id or checkpoint_id required")

    from web.services.deployment_service import deployment_service
    from pipeline.common import PROJECT_ROOT

    run_path = PROJECT_ROOT / "artifacts" / "runs" / run_id
    target_id = str(run_path) if run_path.exists() else run_id

    try:
        dep = deployment_service.deploy_checkpoint(target_id, backend_type="llama.cpp")
        return JSONResponse({"status": "success", "deployment": dep})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=400)

