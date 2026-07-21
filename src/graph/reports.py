"""Deterministic community report generation."""

from __future__ import annotations

import hashlib
from typing import Iterable, List, Mapping

from src.models.graph import (
    CommunityReport,
    GraphCommunity,
    GraphEntity,
    GraphRelationship,
    GraphSourceRef,
)


class CommunityReportBuilder:
    """Build stable, non-LLM community reports."""

    def build(
        self,
        community: GraphCommunity,
        *,
        entities: Iterable[GraphEntity],
        relationships: Iterable[GraphRelationship],
    ) -> CommunityReport:
        entity_by_id = {entity.id: entity for entity in entities}
        relationship_by_id = {
            relationship.id: relationship for relationship in relationships
        }

        selected_entities = [
            entity_by_id[entity_id]
            for entity_id in community.entity_ids
            if entity_id in entity_by_id
        ]
        selected_relationships = [
            relationship_by_id[relationship_id]
            for relationship_id in community.relationship_ids
            if relationship_id in relationship_by_id
        ]

        key_entities = sorted(entity.name for entity in selected_entities)
        key_relationships = sorted(
            relationship.description or relationship.relation_type
            for relationship in selected_relationships
        )
        source_refs = _dedupe_refs(
            [
                ref
                for item in [*selected_entities, *selected_relationships]
                for ref in item.source_refs
            ]
        )
        title = _title(key_entities)
        summary = (
            f"{title} contains {len(selected_entities)} entities and "
            f"{len(selected_relationships)} relationships."
        )
        if key_relationships:
            summary = f"{summary} Key relationship: {key_relationships[0]}."

        return CommunityReport(
            id=stable_report_id(community.id),
            community_id=community.id,
            title=title,
            summary=summary,
            key_entities=key_entities,
            key_relationships=key_relationships,
            source_refs=source_refs,
            metadata={
                "level": community.level,
                "missing_entities": sorted(
                    set(community.entity_ids) - set(entity_by_id)
                ),
                "missing_relationships": sorted(
                    set(community.relationship_ids) - set(relationship_by_id)
                ),
            },
        )

    def build_many(
        self,
        communities: Iterable[GraphCommunity],
        *,
        entities: Iterable[GraphEntity],
        relationships: Iterable[GraphRelationship],
    ) -> List[CommunityReport]:
        entity_list = list(entities)
        relationship_list = list(relationships)
        return [
            self.build(
                community,
                entities=entity_list,
                relationships=relationship_list,
            )
            for community in communities
        ]


def stable_report_id(community_id: str) -> str:
    digest = hashlib.sha1(community_id.encode("utf-8")).hexdigest()
    return f"report_{digest[:16]}"


def _title(entity_names: List[str]) -> str:
    if not entity_names:
        return "Empty Graph Community"
    return "Community: " + ", ".join(entity_names[:3])


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


__all__ = ["CommunityReportBuilder", "stable_report_id"]
