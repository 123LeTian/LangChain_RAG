"""LLM client factory for OpenAI-compatible chat completion models."""

from __future__ import annotations

import os
from typing import Any

from src.chat.model_runtime import ModelRuntimeConfig, resolve_runtime_config


class LLMClientFactory:
    """Create LangChain chat models for default and per-session runtime config."""

    def create_chat_model(
        self,
        model_config: Any = None,
        *,
        temperature: float = 0.1,
        max_tokens: int = 1000,
        streaming: bool = False,
    ) -> Any:
        if model_config is None:
            return self._create_default_chat_model(
                temperature=temperature,
                max_tokens=max_tokens,
                streaming=streaming,
            )

        runtime = (
            model_config
            if isinstance(model_config, ModelRuntimeConfig)
            else resolve_runtime_config(model_config, require_stream=streaming)
        )
        return self._create_runtime_chat_model(
            runtime,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=streaming,
        )

    def create_streaming_chat_model(
        self,
        model_config: Any = None,
        *,
        temperature: float = 0.1,
        max_tokens: int = 1000,
    ) -> Any:
        return self.create_chat_model(
            model_config,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=True,
        )

    def _create_default_chat_model(
        self,
        *,
        temperature: float,
        max_tokens: int,
        streaming: bool,
    ) -> Any:
        from langchain_openai import ChatOpenAI

        kwargs = {
            "model": os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            "api_key": os.getenv("DEEPSEEK_API_KEY"),
            "base_url": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if streaming:
            kwargs["streaming"] = True
        return ChatOpenAI(**kwargs)

    def _create_runtime_chat_model(
        self,
        runtime: ModelRuntimeConfig,
        *,
        temperature: float,
        max_tokens: int,
        streaming: bool,
    ) -> Any:
        from langchain_openai import ChatOpenAI

        kwargs = {
            "model": runtime.model_name,
            "api_key": runtime.api_key or "ollama",
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if runtime.base_url:
            kwargs["base_url"] = runtime.base_url
        if streaming:
            kwargs["streaming"] = True
        return ChatOpenAI(**kwargs)


__all__ = ["LLMClientFactory"]
