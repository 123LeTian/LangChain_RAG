# API Routes — Owner: A

from .chat import router as chat_router
from .evaluation import router as evaluation_router
from .graph import router as graph_router
from .knowledge import router as knowledge_router
from .trace import router as trace_router
from src.api.chat_routes import router as chat_platform_router
from src.api.config_routes import router as config_router

# 所有路由的扁平列表（app.py 中循环注册）
ALL_ROUTERS = [
    chat_router,
    chat_platform_router,
    config_router,
    evaluation_router,
    graph_router,
    knowledge_router,
    trace_router,
]
