"""Graph Index foundation."""

from src.graph.builder import GraphIndexBuilder
from src.graph.community import CommunityDetector
from src.graph.extractor import (
    GraphExtractor,
    MockGraphExtractor,
    RuleBasedGraphExtractor,
)
from src.graph.index import NetworkXGraphIndex
from src.graph.repository import (
    DanglingRelationshipError,
    GraphRepository,
    GraphRepositoryError,
    InMemoryGraphRepository,
)
from src.graph.retriever import (
    GraphGlobalSearchHit,
    GraphGlobalSearchResult,
    GraphLocalSearchResult,
    GraphRetriever,
    GraphSearchHit,
    resolve_graph_scope,
)
from src.graph.reports import CommunityReportBuilder
from src.graph.tool import (
    GraphSearchTool,
    graph_search,
    register_graph_search_tool,
)


__all__ = [
    "CommunityDetector",
    "CommunityReportBuilder",
    "DanglingRelationshipError",
    "GraphExtractor",
    "GraphIndexBuilder",
    "GraphRepository",
    "GraphRepositoryError",
    "GraphGlobalSearchHit",
    "GraphGlobalSearchResult",
    "GraphLocalSearchResult",
    "GraphRetriever",
    "GraphSearchHit",
    "GraphSearchTool",
    "InMemoryGraphRepository",
    "MockGraphExtractor",
    "NetworkXGraphIndex",
    "RuleBasedGraphExtractor",
    "graph_search",
    "register_graph_search_tool",
    "resolve_graph_scope",
]
