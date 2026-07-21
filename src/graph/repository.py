"""In-memory graph repository for Graph Index."""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, Iterable, List, Optional

from src.models.graph import (
    CommunityReport,
    GraphCommunity,
    GraphEntity,
    GraphRelationship,
    GraphSourceRef,
)


class GraphRepositoryError(Exception):
    """Base graph repository error."""


class DanglingRelationshipError(GraphRepositoryError):
    """Raised when a relationship references missing entities."""


class GraphRepository:
    """Repository interface for graph index data."""

    def add_entities(self, kb_id: str, entities: Iterable[GraphEntity]) -> List[GraphEntity]:
        raise NotImplementedError

    def add_relationships(
        self, kb_id: str, relationships: Iterable[GraphRelationship]
    ) -> List[GraphRelationship]:
        raise NotImplementedError

    def get_entity(self, kb_id: str, entity_id: str) -> Optional[GraphEntity]:
        raise NotImplementedError

    def get_relationship(
        self, kb_id: str, relationship_id: str
    ) -> Optional[GraphRelationship]:
        raise NotImplementedError

    def list_entities(self, kb_id: str) -> List[GraphEntity]:
        raise NotImplementedError

    def list_relationships(self, kb_id: str) -> List[GraphRelationship]:
        raise NotImplementedError


class InMemoryGraphRepository(GraphRepository):
    """Per-knowledge-base in-memory graph store."""

    def __init__(self) -> None:
        self._entities: Dict[str, Dict[str, GraphEntity]] = {}
        self._relationships: Dict[str, Dict[str, GraphRelationship]] = {}
        self._communities: Dict[str, Dict[str, GraphCommunity]] = {}
        self._reports: Dict[str, Dict[str, CommunityReport]] = {}

    def add_entities(self, kb_id: str, entities: Iterable[GraphEntity]) -> List[GraphEntity]:
        bucket = self._entities.setdefault(kb_id, {})
        saved = []
        for entity in entities:
            current = bucket.get(entity.id)
            bucket[entity.id] = _merge_entity(current, entity)
            saved.append(deepcopy(bucket[entity.id]))
        return saved

    def add_relationships(
        self, kb_id: str, relationships: Iterable[GraphRelationship]
    ) -> List[GraphRelationship]:
        entity_bucket = self._entities.setdefault(kb_id, {})
        relationship_bucket = self._relationships.setdefault(kb_id, {})
        saved = []
        for relationship in relationships:
            if (
                relationship.source_entity_id not in entity_bucket
                or relationship.target_entity_id not in entity_bucket
            ):
                raise DanglingRelationshipError(
                    f"relationship {relationship.id} references missing entities"
                )
            current = relationship_bucket.get(relationship.id)
            relationship_bucket[relationship.id] = _merge_relationship(
                current, relationship
            )
            saved.append(deepcopy(relationship_bucket[relationship.id]))
        return saved

    def add_communities(
        self, kb_id: str, communities: Iterable[GraphCommunity]
    ) -> List[GraphCommunity]:
        bucket = self._communities.setdefault(kb_id, {})
        saved = []
        for community in communities:
            bucket[community.id] = deepcopy(community)
            saved.append(deepcopy(community))
        return saved

    def add_reports(
        self, kb_id: str, reports: Iterable[CommunityReport]
    ) -> List[CommunityReport]:
        bucket = self._reports.setdefault(kb_id, {})
        saved = []
        for report in reports:
            bucket[report.id] = deepcopy(report)
            saved.append(deepcopy(report))
        return saved

    def get_entity(self, kb_id: str, entity_id: str) -> Optional[GraphEntity]:
        entity = self._entities.get(kb_id, {}).get(entity_id)
        return deepcopy(entity) if entity is not None else None

    def get_relationship(
        self, kb_id: str, relationship_id: str
    ) -> Optional[GraphRelationship]:
        relationship = self._relationships.get(kb_id, {}).get(relationship_id)
        return deepcopy(relationship) if relationship is not None else None

    def get_community(self, kb_id: str, community_id: str) -> Optional[GraphCommunity]:
        community = self._communities.get(kb_id, {}).get(community_id)
        return deepcopy(community) if community is not None else None

    def get_report(self, kb_id: str, report_id: str) -> Optional[CommunityReport]:
        report = self._reports.get(kb_id, {}).get(report_id)
        return deepcopy(report) if report is not None else None

    def list_entities(self, kb_id: str) -> List[GraphEntity]:
        return [deepcopy(item) for item in self._entities.get(kb_id, {}).values()]

    def list_relationships(self, kb_id: str) -> List[GraphRelationship]:
        return [
            deepcopy(item)
            for item in self._relationships.get(kb_id, {}).values()
        ]

    def list_communities(self, kb_id: str) -> List[GraphCommunity]:
        return [deepcopy(item) for item in self._communities.get(kb_id, {}).values()]

    def list_reports(self, kb_id: str) -> List[CommunityReport]:
        return [deepcopy(item) for item in self._reports.get(kb_id, {}).values()]

    def export_graph_data(self, kb_id: str) -> Dict[str, List[dict]]:
        return {
            "entities": [
                entity.model_dump() for entity in self.list_entities(kb_id)
            ],
            "relationships": [
                relationship.model_dump()
                for relationship in self.list_relationships(kb_id)
            ],
            "communities": [
                community.model_dump()
                for community in self.list_communities(kb_id)
            ],
            "reports": [
                report.model_dump() for report in self.list_reports(kb_id)
            ],
        }


def _merge_entity(
    existing: GraphEntity | None, incoming: GraphEntity
) -> GraphEntity:
    if existing is None:
        return deepcopy(incoming)
    refs = _dedupe_refs([*existing.source_refs, *incoming.source_refs])
    metadata = {**existing.metadata, **incoming.metadata}
    description = existing.description or incoming.description
    return existing.model_copy(
        update={
            "description": description,
            "source_refs": refs,
            "metadata": metadata,
        },
        deep=True,
    )


def _merge_relationship(
    existing: GraphRelationship | None, incoming: GraphRelationship
) -> GraphRelationship:
    if existing is None:
        return deepcopy(incoming)
    refs = _dedupe_refs([*existing.source_refs, *incoming.source_refs])
    metadata = {**existing.metadata, **incoming.metadata}
    return existing.model_copy(
        update={
            "weight": existing.weight + incoming.weight,
            "source_refs": refs,
            "metadata": metadata,
        },
        deep=True,
    )


def _dedupe_refs(refs: Iterable[GraphSourceRef]) -> List[GraphSourceRef]:
    seen = set()
    unique = []
    for ref in refs:
        key = (ref.document_id, ref.chunk_id, ref.quote)
        if key in seen:
            continue
        seen.add(key)
        unique.append(ref)
    return unique


__all__ = [
    "DanglingRelationshipError",
    "GraphRepository",
    "GraphRepositoryError",
    "InMemoryGraphRepository",
]
