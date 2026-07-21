"""Graph Index build orchestration."""

from __future__ import annotations

import time
from typing import Any, Iterable, Optional

from src.graph.community import CommunityDetector
from src.graph.extractor import GraphExtractor, RuleBasedGraphExtractor
from src.graph.index import NetworkXGraphIndex
from src.graph.repository import (
    DanglingRelationshipError,
    InMemoryGraphRepository,
)
from src.graph.reports import CommunityReportBuilder
from src.models.graph import GraphBuildResult


class GraphIndexBuilder:
    """Build entity, relationship, community, and report graph artifacts."""

    def __init__(
        self,
        *,
        extractor: Optional[GraphExtractor] = None,
        repository: Optional[InMemoryGraphRepository] = None,
        index: Optional[NetworkXGraphIndex] = None,
        community_detector: Optional[CommunityDetector] = None,
        report_builder: Optional[CommunityReportBuilder] = None,
    ) -> None:
        self.extractor = extractor or RuleBasedGraphExtractor()
        self.repository = repository or InMemoryGraphRepository()
        self.index = index or NetworkXGraphIndex(self.repository)
        self.community_detector = community_detector or CommunityDetector()
        self.report_builder = report_builder or CommunityReportBuilder()

    def build_from_chunks(
        self,
        *,
        kb_id: str,
        chunks: Iterable[Any],
    ) -> GraphBuildResult:
        started = time.perf_counter()
        warnings = []
        chunk_list = list(chunks)
        if not chunk_list:
            warnings.append("no chunks provided")

        extraction = self.extractor.extract_from_chunks(chunk_list)
        warnings.extend(extraction.warnings)

        self.repository.add_entities(kb_id, extraction.entities)
        valid_relationships = [
            relationship
            for relationship in extraction.relationships
            if self.repository.get_entity(kb_id, relationship.source_entity_id)
            and self.repository.get_entity(kb_id, relationship.target_entity_id)
        ]
        dropped = len(extraction.relationships) - len(valid_relationships)
        if dropped:
            warnings.append(f"dropped {dropped} dangling relationship(s)")
        try:
            self.repository.add_relationships(kb_id, valid_relationships)
        except DanglingRelationshipError as exc:
            warnings.append(str(exc))

        graph = self.index.to_networkx(kb_id)
        communities = self.community_detector.detect(graph)
        self.repository.add_communities(kb_id, communities)
        entities = self.repository.list_entities(kb_id)
        relationships = self.repository.list_relationships(kb_id)
        reports = self.report_builder.build_many(
            communities,
            entities=entities,
            relationships=relationships,
        )
        self.repository.add_reports(kb_id, reports)
        duration_ms = round((time.perf_counter() - started) * 1000.0, 3)

        return GraphBuildResult(
            kb_id=kb_id,
            entity_count=len(entities),
            relationship_count=len(relationships),
            community_count=len(communities),
            report_count=len(reports),
            entities=entities,
            relationships=relationships,
            communities=communities,
            reports=reports,
            warnings=warnings,
            duration_ms=duration_ms,
            metadata={
                "chunk_count": len(chunk_list),
                "networkx_node_count": graph.number_of_nodes(),
                "networkx_edge_count": graph.number_of_edges(),
            },
        )


__all__ = ["GraphIndexBuilder"]
