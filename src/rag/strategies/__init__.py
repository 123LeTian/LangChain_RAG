# RAG Strategies — Multi-owner: B, C, D
#
# Each strategy module auto-registers with the StrategyRegistry.
# Import the modules to trigger registration.
#
# C-owned strategies (implemented):
#   from src.rag.strategies import modular     # Owner: C
#   from src.rag.strategies import agentic     # Owner: C
#
# B-owned strategies (to be implemented by B):
#   from src.rag.strategies import naive       # Owner: B
#   from src.rag.strategies import advanced    # Owner: B
#
# D-owned strategies (to be implemented by D):
#   from src.rag.strategies import graph_rag   # Owner: D
#
# Or use auto-discovery:
#   registry.discover_all("src.rag.strategies")

from src.rag.strategies.modular import (
    CompressModule,
    GenerateModule,
    ModularPipeline,
    ModularPipelineBuilder,
    ModularRAGStrategy,
    QueryRewriteModule,
    RerankModule,
    RetrieveModule,
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
    "ModularPipeline",
    "ModularPipelineBuilder",
    "CompressModule",
    "GenerateModule",
    "QueryRewriteModule",
    "RerankModule",
    "RetrieveModule",
    # Agentic (Owner: C)
    "AgenticRAGStrategy",
]
