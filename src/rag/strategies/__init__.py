# RAG Strategies — Multi-owner: B, C, D
#
# Each strategy module auto-registers with the global RAGStrategyRegistry.
# Import the modules to trigger registration.
#
# C-owned strategies (implemented):
#   from src.rag.strategies import modular     — ModularRAGStrategy + ModuleConfig
#   from src.rag.strategies import agentic     — AgenticRAGStrategy
#
# B-owned strategies (to be implemented by B):
#   from src.rag.strategies import naive       — Owner: B
#   from src.rag.strategies import advanced    — Owner: B
#
# D-owned strategy (to be implemented by D):
#   from src.rag.strategies import graph_rag   — Owner: D

from src.rag.strategies.modular import (
    ModuleConfig,
    ModularRAGStrategy,
    validate_module_config,
)
from src.rag.strategies.agentic import AgenticRAGStrategy

# B-owned strategy classes (import when implemented):
# from src.rag.strategies.naive import NaiveRAGStrategy
# from src.rag.strategies.advanced import AdvancedRAGStrategy

# D-owned strategy class (import when implemented):
# from src.rag.strategies.graph_rag import GraphRAGStrategy

__all__ = [
    # Modular (Owner: C)
    "ModularRAGStrategy",
    "ModuleConfig",
    "validate_module_config",
    # Agentic (Owner: C)
    "AgenticRAGStrategy",
]
