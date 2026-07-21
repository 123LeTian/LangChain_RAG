# API Routes — Owner: A

from .chat import router as chat_router
from .knowledge import router as knowledge_router

# 所有路由的扁平列表（app.py 中循环注册）
ALL_ROUTERS = [
    chat_router,
    knowledge_router,
]
