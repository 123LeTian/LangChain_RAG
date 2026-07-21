# RAG Strategies — Multi-owner: B, C, D
#
# Each strategy module auto-registers with the global RAGStrategyRegistry.
# Import the modules to trigger registration.

from .advanced import AdvancedRAGStrategy
from .naive import NaiveRAGStrategy

from src.rag.strategies.modular import (
    ModuleConfig,
    ModularRAGStrategy,
    validate_module_config,
)
from src.rag.strategies.agentic import AgenticRAGStrategy

# D-owned strategy class (import when implemented):
# from src.rag.strategies.graph_rag import GraphRAGStrategy

__all__ = [
    # B-owned
    "AdvancedRAGStrategy",
    "NaiveRAGStrategy",
    # Modular (Owner: C)
    "ModularRAGStrategy",
    "ModuleConfig",
    "validate_module_config",
    # Agentic (Owner: C)
    "AgenticRAGStrategy",
]
