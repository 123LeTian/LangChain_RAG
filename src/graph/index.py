"""NetworkX-backed graph index."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import networkx as nx

from src.graph.extractor import GraphExtractor, RuleBasedGraphExtractor
from src.graph.repository import InMemoryGraphRepository
from src.models.graph import GraphEntity, GraphRelationship


class NetworkXGraphIndex:
    """Build and expose a NetworkX graph from repository data."""

    def __init__(self, repository: Optional[InMemoryGraphRepository] = None) -> None:
        self.repository = repository or InMemoryGraphRepository()

    def build_from_chunks(
        self,
        kb_id: str,
        chunks: Iterable[Any],
        *,
        extractor: Optional[GraphExtractor] = None,
    ) -> nx.Graph:
        active_extractor = extractor or RuleBasedGraphExtractor()
        result = active_extractor.extract_from_chunks(chunks)
        self.repository.add_entities(kb_id, result.entities)
        self.repository.add_relationships(kb_id, result.relationships)
        return self.to_networkx(kb_id)

    def to_networkx(self, kb_id: str) -> nx.Graph:
        graph = nx.Graph(kb_id=kb_id)
        for entity in self.repository.list_entities(kb_id):
            graph.add_node(
                entity.id,
                entity=entity,
                name=entity.name,
                type=entity.type,
                description=entity.description,
                source_refs=[ref.model_dump() for ref in entity.source_refs],
                metadata=dict(entity.metadata),
            )

        for relationship in self.repository.list_relationships(kb_id):
            if (
                relationship.source_entity_id not in graph
                or relationship.target_entity_id not in graph
            ):
                continue
            existing = graph.get_edge_data(
                relationship.source_entity_id,
                relationship.target_entity_id,
                default={},
            )
            relationship_ids = list(existing.get("relationship_ids", []))
            relationships = list(existing.get("relationships", []))
            relationship_ids.append(relationship.id)
            relationships.append(relationship)
            graph.add_edge(
                relationship.source_entity_id,
                relationship.target_entity_id,
                weight=float(existing.get("weight", 0.0)) + relationship.weight,
                relationship_ids=relationship_ids,
                relationships=relationships,
                source_refs=[
                    ref.model_dump() for ref in relationship.source_refs
                ],
                relation_types=sorted(
                    {
                        *existing.get("relation_types", []),
                        relationship.relation_type,
                    }
                ),
            )
        return graph

    def from_networkx(self, kb_id: str, graph: nx.Graph) -> None:
        entities: List[GraphEntity] = []
        relationships: List[GraphRelationship] = []
        for _, data in graph.nodes(data=True):
            entity = data.get("entity")
            if isinstance(entity, GraphEntity):
                entities.append(entity)
        for _, _, data in graph.edges(data=True):
            for relationship in data.get("relationships", []):
                if isinstance(relationship, GraphRelationship):
                    relationships.append(relationship)
        self.repository.add_entities(kb_id, entities)
        self.repository.add_relationships(kb_id, relationships)

    def export_graph_data(self, kb_id: str) -> Dict[str, Any]:
        graph = self.to_networkx(kb_id)
        return {
            **self.repository.export_graph_data(kb_id),
            "node_count": graph.number_of_nodes(),
            "edge_count": graph.number_of_edges(),
        }


__all__ = ["NetworkXGraphIndex"]
