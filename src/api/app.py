"""
API Application — Owner: A
FastAPI application entry point.
禁止放 RAG 算法逻辑。
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .errors import register_error_handlers
from .routes import ALL_ROUTERS


# ============================================================================
# 应用生命周期
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时初始化，关闭时清理资源"""
    # 启动逻辑：后续可在这里初始化数据库连接池、模型预热等
    yield
    # 关闭逻辑：后续可在这里关闭连接、清理临时文件等


# ============================================================================
# FastAPI 实例
# ============================================================================

app = FastAPI(
    title="LangChain RAG API",
    description="多范式 RAG 实验平台后端 — Naive / Advanced / Modular / Graph / Agentic",
    version="0.1.0",
    lifespan=lifespan,
)

# ============================================================================
# CORS — 允许前端开发服务器跨域请求
# ============================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],  # Vite 默认地址
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# 注册统一异常处理器
# ============================================================================

register_error_handlers(app)

# ============================================================================
# 注册所有路由
# ============================================================================

for router in ALL_ROUTERS:
    app.include_router(router)


# ============================================================================
# 健康检查端点 — 前后端连通性验证
# ============================================================================

@app.get("/health")
async def health_check():
    """健康检查 — A1 阶段门禁：前后端都能启动后通过此端点验证"""
    return {"status": "ok", "version": "0.1.0", "service": "LangChain RAG API"}
