"""
web/routers/inference.py
------------------------
OpenAI-compatible inference API endpoints (/v1/chat/completions, /v1/models).
Enables external applications, agents, Cursor, and LangChain to interface directly
with DocTune fine-tuned models.
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from web.services.inference_service import inference_service

router = APIRouter(tags=["Inference API"])


class ChatMessage(BaseModel):
    role: str
    content: str
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str = Field(default="doctune-merged", description="ID of the model or checkpoint to use")
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    max_tokens: Optional[int] = 512
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0


@router.post("/v1/chat/completions")
async def create_chat_completion(payload: ChatCompletionRequest):
    """
    OpenAI-compatible chat completion endpoint.
    Supports both standard JSON responses and SSE streaming (stream=True).
    """
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

    completion = inference_service.generate_completion(
        messages=dict_messages,
        model_name=payload.model,
        temperature=payload.temperature or 0.7,
        max_tokens=payload.max_tokens or 512,
    )
    return JSONResponse(completion)


@router.get("/v1/models")
async def list_models():
    """List available fine-tuned models and base models in OpenAI schema."""
    models = inference_service.list_available_models()
    return JSONResponse({
        "object": "list",
        "data": models,
    })


@router.get("/v1/models/{model_id}")
async def retrieve_model(model_id: str):
    """Retrieve details for a specific model ID."""
    models = inference_service.list_available_models()
    found = next((m for m in models if m["id"] == model_id), None)
    if not found:
        # Fallback dynamic object
        return JSONResponse({
            "id": model_id,
            "object": "model",
            "created": 1700000000,
            "owned_by": "doctune",
        })
    return JSONResponse(found)
