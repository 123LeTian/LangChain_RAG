"""Community detection for graph index MVP."""

from __future__ import annotations

import hashlib
from typing import List

import networkx as nx

from src.models.graph import GraphCommunity


class CommunityDetector:
    """Detect level-0 communities using connected components."""

    def detect(self, graph: nx.Graph, *, level: int = 0) -> List[GraphCommunity]:
        if graph.number_of_nodes() == 0:
            return []

        communities = []
        for component in nx.connected_components(graph):
            entity_ids = sorted(str(entity_id) for entity_id in component)
            relationship_ids = set()
            for source_id, target_id in graph.subgraph(component).edges():
                data = graph.get_edge_data(source_id, target_id, default={})
                relationship_ids.update(data.get("relationship_ids", []))
            sorted_relationship_ids = sorted(relationship_ids)
            community_id = stable_community_id(entity_ids, level=level)
            communities.append(
                GraphCommunity(
                    id=community_id,
                    level=level,
                    entity_ids=entity_ids,
                    relationship_ids=sorted_relationship_ids,
                    summary=(
                        f"Community with {len(entity_ids)} entities and "
                        f"{len(sorted_relationship_ids)} relationships."
                    ),
                    metadata={"algorithm": "connected_components"},
                )
            )
        return sorted(communities, key=lambda community: community.id)


def stable_community_id(entity_ids: List[str], *, level: int = 0) -> str:
    raw = f"{level}:" + ",".join(sorted(entity_ids))
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()
    return f"comm_{digest[:16]}"


__all__ = ["CommunityDetector", "stable_community_id"]
