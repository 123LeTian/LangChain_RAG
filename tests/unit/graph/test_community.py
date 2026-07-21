import networkx as nx

from src.graph.community import CommunityDetector


def test_empty_graph_has_no_communities():
    communities = CommunityDetector().detect(nx.Graph())

    assert communities == []


def test_single_node_graph_has_one_community():
    graph = nx.Graph()
    graph.add_node("ent-1")

    communities = CommunityDetector().detect(graph)

    assert len(communities) == 1
    assert communities[0].entity_ids == ["ent-1"]
    assert communities[0].relationship_ids == []


def test_connected_components_create_stable_communities():
    graph = nx.Graph()
    graph.add_edge("ent-1", "ent-2", relationship_ids=["rel-1"])
    graph.add_edge("ent-3", "ent-4", relationship_ids=["rel-2"])

    communities = CommunityDetector().detect(graph)

    assert len(communities) == 2
    assert {tuple(community.entity_ids) for community in communities} == {
        ("ent-1", "ent-2"),
        ("ent-3", "ent-4"),
    }
    assert {tuple(community.relationship_ids) for community in communities} == {
        ("rel-1",),
        ("rel-2",),
    }
