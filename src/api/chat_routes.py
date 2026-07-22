"""Chat session and message routes."""

from __future__ import annotations

import json
from typing import List, Optional

from fastapi import APIRouter, Body, Response
from sse_starlette.sse import EventSourceResponse

from src.api.dependencies import (
    ChatApplicationServiceDep,
    ChatExportServiceDep,
    ChatMessageServiceDep,
    ChatSearchServiceDep,
    ChatSessionServiceDep,
    ChatTokenServiceDep,
    ModelRegistryDep,
    PresetServiceDep,
)
from src.api.errors import NotFoundError, ValidationError
from src.chat.schemas import (
    ChatMessage,
    ChatMessageCreate,
    ChatExportResponse,
    ChatPresetCreate,
    ChatPresetUpdate,
    ChatSearchResponse,
    ChatSession,
    ChatSessionCreate,
    ChatSessionModelUpdate,
    ChatSessionPresetUpdate,
    ChatSessionStats,
    ChatStats,
    ChatSessionUpdate,
    ChatStreamRequest,
)
from src.chat.model_registry import ModelNotFoundError
from src.chat.preset_service import (
    ChatPresetNotFoundError,
    ChatPresetReadOnlyError,
)
from src.chat.search_service import ChatSearchValidationError
from src.chat.session_service import ChatSessionNotFoundError


router = APIRouter(prefix="/api/chat", tags=["Chat Sessions"])


def _not_found(session_id: str) -> NotFoundError:
    return NotFoundError(f"Chat session {session_id} not found")


@router.get("/models")
async def list_models(
    registry: ModelRegistryDep = None,  # type: ignore
) -> dict:
    return registry.public_response()


@router.get("/presets")
async def list_presets(
    service: PresetServiceDep = None,  # type: ignore
) -> dict:
    return service.public_response()


@router.get("/search", response_model=ChatSearchResponse)
async def search_messages(
    q: str,
    session_id: Optional[str] = None,
    role: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
    service: ChatSearchServiceDep = None,  # type: ignore
) -> ChatSearchResponse:
    try:
        return service.search(
            q,
            session_id=session_id,
            role=role,
            limit=limit,
            offset=offset,
        )
    except ChatSearchValidationError as exc:
        raise ValidationError(str(exc))


@router.get("/stats", response_model=ChatStats)
async def global_stats(
    service: ChatTokenServiceDep = None,  # type: ignore
) -> ChatStats:
    return service.global_stats()


@router.post("/presets", status_code=201)
async def create_preset(
    request: ChatPresetCreate,
    service: PresetServiceDep = None,  # type: ignore
) -> dict:
    return service.create_user_preset(request).public_dict(include_prompt=True)


@router.patch("/presets/{preset_id}")
async def update_preset(
    preset_id: str,
    request: ChatPresetUpdate,
    service: PresetServiceDep = None,  # type: ignore
) -> dict:
    try:
        return service.update_user_preset(
            preset_id,
            request,
        ).public_dict(include_prompt=True)
    except ChatPresetReadOnlyError:
        raise ValidationError(f"Preset {preset_id} is read-only")
    except ChatPresetNotFoundError:
        raise NotFoundError(f"Preset {preset_id} not found")


@router.delete("/presets/{preset_id}", status_code=204)
async def delete_preset(
    preset_id: str,
    service: PresetServiceDep = None,  # type: ignore
) -> Response:
    try:
        service.delete_user_preset(preset_id)
    except ChatPresetReadOnlyError:
        raise ValidationError(f"Preset {preset_id} is read-only")
    except ChatPresetNotFoundError:
        raise NotFoundError(f"Preset {preset_id} not found")
    return Response(status_code=204)


@router.post("/sessions", response_model=ChatSession, status_code=201)
async def create_session(
    request: Optional[ChatSessionCreate] = Body(default=None),
    service: ChatSessionServiceDep = None,  # type: ignore
) -> ChatSession:
    return service.create_session(request)


@router.get("/sessions", response_model=List[ChatSession])
async def list_sessions(
    service: ChatSessionServiceDep = None,  # type: ignore
) -> List[ChatSession]:
    return service.list_sessions()


@router.get("/sessions/{session_id}", response_model=ChatSession)
async def get_session(
    session_id: str,
    service: ChatSessionServiceDep = None,  # type: ignore
) -> ChatSession:
    try:
        return service.get_session(session_id)
    except ChatSessionNotFoundError:
        raise _not_found(session_id)


@router.patch("/sessions/{session_id}", response_model=ChatSession)
async def update_session(
    session_id: str,
    request: ChatSessionUpdate,
    service: ChatSessionServiceDep = None,  # type: ignore
    registry: ModelRegistryDep = None,  # type: ignore
    preset_service: PresetServiceDep = None,  # type: ignore
) -> ChatSession:
    try:
        if "model_id" in request.model_fields_set and request.model_id is not None:
            request.model_id = registry.get_enabled(request.model_id).id
        if "preset_id" in request.model_fields_set and request.preset_id is not None:
            request.preset_id = preset_service.resolve(request.preset_id).id
        return service.update_session(session_id, request)
    except ChatSessionNotFoundError:
        raise _not_found(session_id)
    except ModelNotFoundError:
        raise ValidationError(f"Model {request.model_id} is not available")
    except ChatPresetNotFoundError:
        raise NotFoundError(f"Preset {request.preset_id} not found")


@router.delete("/sessions/{session_id}", status_code=204)
async def delete_session(
    session_id: str,
    service: ChatSessionServiceDep = None,  # type: ignore
) -> Response:
    try:
        service.delete_session(session_id)
    except ChatSessionNotFoundError:
        raise _not_found(session_id)
    return Response(status_code=204)


@router.get("/sessions/{session_id}/messages", response_model=List[ChatMessage])
async def list_messages(
    session_id: str,
    service: ChatMessageServiceDep = None,  # type: ignore
) -> List[ChatMessage]:
    try:
        return service.list_messages(session_id)
    except ChatSessionNotFoundError:
        raise _not_found(session_id)


@router.get("/sessions/{session_id}/stats", response_model=ChatSessionStats)
async def session_stats(
    session_id: str,
    service: ChatTokenServiceDep = None,  # type: ignore
) -> ChatSessionStats:
    try:
        return service.session_stats(session_id)
    except ChatSessionNotFoundError:
        raise _not_found(session_id)


@router.get("/sessions/{session_id}/export", response_model=ChatExportResponse)
async def export_session(
    session_id: str,
    service: ChatExportServiceDep = None,  # type: ignore
) -> ChatExportResponse:
    try:
        return service.export_session(session_id)
    except ChatSessionNotFoundError:
        raise _not_found(session_id)


@router.post("/sessions/{session_id}/messages", response_model=ChatMessage, status_code=201)
async def save_message(
    session_id: str,
    request: ChatMessageCreate,
    service: ChatMessageServiceDep = None,  # type: ignore
) -> ChatMessage:
    try:
        return service.save_message(session_id, request)
    except ChatSessionNotFoundError:
        raise _not_found(session_id)


@router.patch("/sessions/{session_id}/model", response_model=ChatSession)
async def update_session_model(
    session_id: str,
    request: ChatSessionModelUpdate,
    session_service: ChatSessionServiceDep = None,  # type: ignore
    registry: ModelRegistryDep = None,  # type: ignore
) -> ChatSession:
    try:
        model = registry.get_enabled(request.model_id)
        return session_service.update_model(session_id, model.id)
    except ChatSessionNotFoundError:
        raise _not_found(session_id)
    except ModelNotFoundError:
        raise ValidationError(f"Model {request.model_id} is not available")


@router.patch("/sessions/{session_id}/preset", response_model=ChatSession)
async def update_session_preset(
    session_id: str,
    request: ChatSessionPresetUpdate,
    session_service: ChatSessionServiceDep = None,  # type: ignore
    preset_service: PresetServiceDep = None,  # type: ignore
) -> ChatSession:
    try:
        preset = preset_service.resolve(request.preset_id)
        return session_service.update_preset(session_id, preset.id)
    except ChatSessionNotFoundError:
        raise _not_found(session_id)
    except ChatPresetNotFoundError:
        raise NotFoundError(f"Preset {request.preset_id} not found")


@router.post("/sessions/{session_id}/stream")
async def stream_session(
    session_id: str,
    request: ChatStreamRequest,
    service: ChatApplicationServiceDep = None,  # type: ignore
):
    try:
        service.ensure_stream_options(
            session_id,
            model_id=request.model_id,
            preset_id=request.preset_id,
        )
    except ChatSessionNotFoundError:
        raise _not_found(session_id)
    except ModelNotFoundError:
        raise ValidationError(f"Model {request.model_id} is not available")
    except ChatPresetNotFoundError:
        raise NotFoundError(f"Preset {request.preset_id} not found")

    async def event_generator():
        async for event in service.stream_session(session_id, request):
            event_type = event.get("type", "message")
            yield {
                "event": event_type,
                "data": json.dumps(event, ensure_ascii=False),
            }

    return EventSourceResponse(event_generator())
