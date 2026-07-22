"""Pydantic models for chat sessions and messages."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


ChatMessageRole = Literal["user", "assistant", "system"]
ChatPresetOwnerType = Literal["system", "user"]
DEFAULT_CHAT_TITLE = "新对话"


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""

    return datetime.now(timezone.utc)


class ChatSession(BaseModel):
    id: str
    title: str = DEFAULT_CHAT_TITLE
    model_id: Optional[str] = None
    preset_id: Optional[str] = None
    rag_mode: Optional[str] = None
    knowledge_base_id: Optional[str] = None
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class ChatSessionCreate(BaseModel):
    title: str = DEFAULT_CHAT_TITLE
    model_id: Optional[str] = None
    preset_id: Optional[str] = None
    rag_mode: Optional[str] = None
    knowledge_base_id: Optional[str] = None

    @field_validator("title")
    @classmethod
    def title_must_not_be_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("title cannot be empty")
        return stripped


class ChatSessionUpdate(BaseModel):
    title: Optional[str] = None
    model_id: Optional[str] = None
    preset_id: Optional[str] = None
    rag_mode: Optional[str] = None
    knowledge_base_id: Optional[str] = None

    @field_validator("title", "model_id", "preset_id", "rag_mode", "knowledge_base_id")
    @classmethod
    def optional_text_must_not_be_blank(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        stripped = value.strip()
        if not stripped:
            raise ValueError("field cannot be empty")
        return stripped


class ChatMessage(BaseModel):
    id: str
    session_id: str
    role: ChatMessageRole
    content: str
    citations: Optional[List[Any]] = None
    trace: Optional[List[Any]] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: Optional[int] = None
    created_at: datetime = Field(default_factory=utc_now)


class ChatMessageCreate(BaseModel):
    role: ChatMessageRole
    content: str = Field(..., min_length=1)
    citations: Optional[List[Any]] = None
    trace: Optional[List[Any]] = None
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    latency_ms: Optional[int] = Field(default=None, ge=0)

    @field_validator("content")
    @classmethod
    def content_must_not_be_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("content cannot be empty")
        return stripped


class ChatStreamRequest(BaseModel):
    question: str = Field(..., min_length=1)
    rag_mode: Optional[str] = None
    knowledge_base_id: Optional[str] = None
    model_id: Optional[str] = None
    preset_id: Optional[str] = None
    top_k: Optional[int] = Field(default=None, ge=1)
    rerank_top_k: Optional[int] = Field(default=None, ge=1)
    score_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    rewrite_enabled: Optional[bool] = None
    retrieve_enabled: Optional[bool] = None
    rerank_enabled: Optional[bool] = None
    compress_enabled: Optional[bool] = None
    verify_enabled: Optional[bool] = None

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("question cannot be empty")
        return stripped


class ChatSessionModelUpdate(BaseModel):
    model_id: str = Field(..., min_length=1)

    @field_validator("model_id")
    @classmethod
    def model_id_must_not_be_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("model_id cannot be empty")
        return stripped


class ChatSessionPresetUpdate(BaseModel):
    preset_id: str = Field(..., min_length=1)

    @field_validator("preset_id")
    @classmethod
    def preset_id_must_not_be_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("preset_id cannot be empty")
        return stripped


class ChatModelCreate(BaseModel):
    id: Optional[str] = None
    provider: str = Field(..., min_length=1)
    display_name: str = Field(..., min_length=1)
    model_name: str = Field(..., min_length=1)
    base_url: Optional[str] = None
    api_key_env: Optional[str] = None
    description: str = ""
    supports_stream: bool = True
    supports_tools: bool = False
    supports_vision: bool = False
    enabled: bool = True

    @field_validator("id", "provider", "display_name", "model_name", "base_url", "api_key_env")
    @classmethod
    def optional_text_must_not_be_blank(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        stripped = value.strip()
        if not stripped:
            raise ValueError("field cannot be empty")
        return stripped


class ChatModelUpdate(BaseModel):
    provider: Optional[str] = None
    display_name: Optional[str] = None
    model_name: Optional[str] = None
    base_url: Optional[str] = None
    api_key_env: Optional[str] = None
    description: Optional[str] = None
    supports_stream: Optional[bool] = None
    supports_tools: Optional[bool] = None
    supports_vision: Optional[bool] = None
    enabled: Optional[bool] = None

    @field_validator("provider", "display_name", "model_name", "base_url", "api_key_env")
    @classmethod
    def optional_text_must_not_be_blank(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        stripped = value.strip()
        if not stripped:
            raise ValueError("field cannot be empty")
        return stripped


class ChatModelDefaultUpdate(BaseModel):
    model_id: str = Field(..., min_length=1)

    @field_validator("model_id")
    @classmethod
    def model_id_must_not_be_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("model_id cannot be empty")
        return stripped


class ChatPreset(BaseModel):
    id: str
    name: str
    description: str = ""
    system_prompt: str
    rag_prompt_hint: Optional[str] = None
    owner_type: ChatPresetOwnerType = "user"
    is_default: bool = False
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def public_dict(self, *, include_prompt: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "owner_type": self.owner_type,
            "is_default": self.is_default,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
        if include_prompt:
            payload["system_prompt"] = self.system_prompt
            payload["rag_prompt_hint"] = self.rag_prompt_hint
        return payload


class ChatPresetCreate(BaseModel):
    name: str = Field(..., min_length=1)
    description: str = ""
    system_prompt: str = Field(..., min_length=1)
    rag_prompt_hint: Optional[str] = None

    @field_validator("name", "system_prompt")
    @classmethod
    def required_text_must_not_be_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("field cannot be empty")
        return stripped


class ChatPresetUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    system_prompt: Optional[str] = None
    rag_prompt_hint: Optional[str] = None

    @field_validator("name", "system_prompt")
    @classmethod
    def optional_required_text_must_not_be_blank(
        cls,
        value: Optional[str],
    ) -> Optional[str]:
        if value is None:
            return value
        stripped = value.strip()
        if not stripped:
            raise ValueError("field cannot be empty")
        return stripped


class ChatSearchItem(BaseModel):
    session_id: str
    session_title: str
    message_id: str
    role: ChatMessageRole
    snippet: str
    created_at: datetime


class ChatSearchResponse(BaseModel):
    items: List[ChatSearchItem]
    total: int


class ChatStats(BaseModel):
    sessions_count: int
    messages_count: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatSessionStats(BaseModel):
    session_id: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    messages_count: int


class ChatExportResponse(BaseModel):
    filename: str
    content: str
