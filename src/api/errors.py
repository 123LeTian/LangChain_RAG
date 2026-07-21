"""
API Errors — Owner: A
集中式错误处理与异常定义。
所有路由抛出的异常都通过此模块转换为统一 JSON 格式。
"""

from __future__ import annotations

from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse


# ============================================================================
# 自定义业务异常 — 统一错误格式
# ============================================================================

class APIError(Exception):
    """API 层统一异常基类，子类只需指定 status_code 与 code"""

    status_code: int = 500
    code: str = "internal_error"
    message: str = "Internal server error"

    def __init__(self, message: str | None = None, detail: dict[str, Any] | None = None):
        self.message = message or self.message
        self.detail = detail or {}


class NotFoundError(APIError):
    """资源不存在：知识库、文档、评测运行等"""
    status_code = 404
    code = "not_found"
    message = "Resource not found"


class ValidationError(APIError):
    """请求参数校验失败"""
    status_code = 422
    code = "validation_error"
    message = "Validation failed"


class KnowledgeBaseError(APIError):
    """知识库操作错误：索引失败、文档格式不支持等"""
    status_code = 400
    code = "knowledge_base_error"
    message = "Knowledge base operation failed"


class RAGError(APIError):
    """RAG 执行错误：策略不可用、工具调用失败等"""
    status_code = 500
    code = "rag_error"
    message = "RAG execution failed"


class TimeoutError(APIError):
    """请求超时（RAG 或 Agent 步数/时间超限）"""
    status_code = 504
    code = "timeout"
    message = "Request timed out"


# ============================================================================
# FastAPI 异常处理器 — 统一 JSON 响应
# ============================================================================

async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
    """将所有 APIError 子类转为统一 JSON 格式"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.code,
                "message": exc.message,
                "detail": exc.detail,
            }
        },
    )


def register_error_handlers(app):
    """在 FastAPI app 上注册所有异常处理器"""
    # 注册所有 APIError 子类（FastAPI 可以匹配继承关系，只需注册基类）
    app.add_exception_handler(APIError, api_error_handler)
