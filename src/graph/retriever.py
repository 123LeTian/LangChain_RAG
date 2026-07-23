"""Graph local/global retrieval over the D3 graph index."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence

from pydantic import BaseModel, Field

from src.graph.repository import InMemoryGraphRepository
from src.models.graph import (
    CommunityReport,
    GraphEntity,
    GraphRelationship,
    GraphSourceRef,
)
from src.models.rag import RAGChunk, RAGContext, RAGSource


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "is",
    "of",
    "the",
    "to",
    "what",
    "how",
    "which",
    "who",
    "me",
    "about",
    "explain",
    "summarize",
    "overview",
    "\u54ea\u4e9b",
    "\u4ec0\u4e48",
    "\u5173\u7cfb",
    "\u4e4b\u95f4",
    "\u5206\u522b",
    "\u5b83\u4eec",
    "\u8bf7",
}

GLOBAL_SCOPE_HINTS = {
    "整体",
    "总体",
    "主题",
    "趋势",
    "概括",
    "总结",
    "经营重点",
    "经营优势",
    "重点板块",
    "板块",
    "主线",
    "逻辑",
    "归纳",
    "支撑",
    "长期发展",
    "说明",
    "体现",
    "意味着",
    "成果",
    "影响",
    "表明",
    "summarize",
    "summary",
    "overall",
    "global",
    "theme",
    "trend",
    "overview",
}


class GraphSearchHit(BaseModel):
    """Local graph search hit centered on one matched entity."""

    entity_id: str
    entity_name: str
    score: float
    matched_text: str
    relationships: List[GraphRelationship] = Field(default_factory=list)
    neighbor_entities: List[GraphEntity] = Field(default_factory=list)
    source_refs: List[GraphSourceRef] = Field(default_factory=list)
    path: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GraphGlobalSearchHit(BaseModel):
    """Global graph search hit backed by a CommunityReport."""

    community_id: str
    report_id: str
    title: str
    summary: str
    score: float
    key_entities: List[str] = Field(default_factory=list)
    key_relationships: List[str] = Field(default_factory=list)
    source_refs: List[GraphSourceRef] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GraphLocalSearchResult(BaseModel):
    """Local graph search result."""

    query: str
    kb_id: str
    hits: List[GraphSearchHit] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GraphGlobalSearchResult(BaseModel):
    """Global graph search result."""

    query: str
    kb_id: str
    hits: List[GraphGlobalSearchHit] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GraphRetriever:
    """Keyword MVP retriever for D4 local and global GraphRAG search."""

    retriever_name = "graph_retriever"

    def __init__(self, repository: InMemoryGraphRepository) -> None:
        self.repository = repository

    def local_search(
        self, query: str, *, kb_id: str, top_k: int = 5
    ) -> GraphLocalSearchResult:
        warnings = _validate_query(query, kb_id, top_k)
        if warnings:
            return GraphLocalSearchResult(query=query, kb_id=kb_id, warnings=warnings)

        entities = self.repository.list_entities(kb_id)
        relationships = self.repository.list_relationships(kb_id)
        if not entities:
            return GraphLocalSearchResult(
                query=query,
                kb_id=kb_id,
                warnings=[f"knowledge graph for kb_id '{kb_id}' is empty"],
            )

        tokens = _query_tokens(query)
        intent = _query_intent(query)
        hits = []
        for entity in entities:
            score, matched_text, reason = _entity_score(entity, tokens, query)
            related = [
                relationship
                for relationship in relationships
                if relationship.source_entity_id == entity.id
                or relationship.target_entity_id == entity.id
            ]
            relationship_score, relationship_reason, relationship_match = _relationships_score(
                related,
                tokens,
                query,
            )
            score += relationship_score
            if relationship_reason:
                reason = "; ".join(part for part in [reason, relationship_reason] if part)
                matched_text = matched_text or relationship_match
            intent_score, intent_reason = _intent_score(entity, related, intent)
            score += intent_score
            if intent_reason:
                reason = "; ".join(part for part in [reason, intent_reason] if part)
            if score <= 0:
                continue
            neighbors = _neighbor_entities(entity, related, entities)
            source_refs = _dedupe_source_refs(
                [
                    *entity.source_refs,
                    *[
                        ref
                        for relationship in related
                        for ref in relationship.source_refs
                    ],
                    *[
                        ref
                        for neighbor in neighbors
                        for ref in neighbor.source_refs
                    ],
                ]
            )
            score += sum(relationship.weight for relationship in related) * 0.1
            score += min(len(source_refs), 5) * 0.05
            hits.append(
                GraphSearchHit(
                    entity_id=entity.id,
                    entity_name=entity.name,
                    score=round(score, 6),
                    matched_text=matched_text,
                    relationships=related,
                    neighbor_entities=neighbors,
                    source_refs=source_refs,
                    path=_local_path(query, entity, related, neighbors, source_refs),
                    metadata={
                        "graph_scope": "local",
                        "score_reason": reason,
                        "relationship_count": len(related),
                    },
                )
            )

        hits.sort(
            key=lambda hit: (
                hit.score,
                len(hit.relationships),
                len(hit.source_refs),
                hit.entity_name.lower(),
            ),
            reverse=True,
        )
        warnings = [] if hits else ["no matching graph entities found"]
        return GraphLocalSearchResult(
            query=query,
            kb_id=kb_id,
            hits=hits[:top_k],
            warnings=warnings,
            metadata={
                "entity_count": len(entities),
                "relationship_count": len(relationships),
            },
        )

    def global_search(
        self, query: str, *, kb_id: str, top_k: int = 5
    ) -> GraphGlobalSearchResult:
        warnings = _validate_query(query, kb_id, top_k)
        if warnings:
            return GraphGlobalSearchResult(query=query, kb_id=kb_id, warnings=warnings)

        reports = self.repository.list_reports(kb_id)
        if not reports:
            return GraphGlobalSearchResult(
                query=query,
                kb_id=kb_id,
                warnings=[f"knowledge graph for kb_id '{kb_id}' has no community reports"],
            )

        tokens = _query_tokens(query)
        hits = []
        for report in reports:
            score, reason = _report_score(report, tokens, query)
            if score <= 0:
                continue
            hits.append(
                GraphGlobalSearchHit(
                    community_id=report.community_id,
                    report_id=report.id,
                    title=report.title,
                    summary=report.summary,
                    score=round(score + min(len(report.source_refs), 5) * 0.05, 6),
                    key_entities=list(report.key_entities),
                    key_relationships=list(report.key_relationships),
                    source_refs=list(report.source_refs),
                    metadata={
                        "graph_scope": "global",
                        "score_reason": reason,
                    },
                )
            )

        hits.sort(
            key=lambda hit: (hit.score, len(hit.source_refs), hit.title.lower()),
            reverse=True,
        )
        if not hits:
            fallback_reports = [
                report
                for report in sorted(
                    reports,
                    key=lambda item: (
                        _report_quality_score(item),
                        len(item.source_refs),
                        len(item.key_relationships),
                        len(item.key_entities),
                    ),
                    reverse=True,
                )
                if _report_quality_score(report) > 0
            ]
            hits = [
                GraphGlobalSearchHit(
                    community_id=report.community_id,
                    report_id=report.id,
                    title=report.title,
                    summary=report.summary,
                    score=round(
                        0.1
                        + _report_quality_score(report) * 0.01
                        + min(len(report.source_refs), 5) * 0.01,
                        6,
                    ),
                    key_entities=list(report.key_entities),
                    key_relationships=list(report.key_relationships),
                    source_refs=list(report.source_refs),
                    metadata={
                        "graph_scope": "global",
                        "score_reason": "global_fallback_no_keyword_match",
                        "quality_score": _report_quality_score(report),
                    },
                )
                for report in fallback_reports[:top_k]
            ]
        warnings = [] if hits else ["no matching community reports found"]
        return GraphGlobalSearchResult(
            query=query,
            kb_id=kb_id,
            hits=hits[:top_k],
            warnings=warnings,
            metadata={"report_count": len(reports)},
        )

    def graph_search(
        self,
        query: str,
        top_k: int = 5,
        **kwargs: Any,
    ) -> RAGContext:
        """Compatibility method for GraphRetrieverProtocol."""

        kb_id = str(kwargs.get("kb_id") or "default")
        scope = str(kwargs.get("scope") or kwargs.get("graph_scope") or "local")
        if scope == "global":
            result = self.global_search(query, kb_id=kb_id, top_k=top_k)
            chunks = [
                _rag_chunk_from_global_hit(hit, rank=index + 1)
                for index, hit in enumerate(result.hits)
            ]
            warnings = result.warnings
        else:
            result = self.local_search(query, kb_id=kb_id, top_k=top_k)
            chunks = [
                _rag_chunk_from_local_hit(hit, rank=index + 1)
                for index, hit in enumerate(result.hits)
            ]
            warnings = result.warnings

        return RAGContext(
            query=query,
            chunks=chunks,
            retrieval_method=f"graph_{scope}",
            total_candidates=len(chunks),
            metadata={"kb_id": kb_id, "graph_scope": scope, "warnings": warnings},
        )

    def get_entity(self, entity_id: str, *, kb_id: str = "default") -> Optional[Dict[str, Any]]:
        entity = self.repository.get_entity(kb_id, entity_id)
        return entity.model_dump() if entity is not None else None

    def get_community_report(
        self, community_id: str, *, kb_id: str = "default"
    ) -> Optional[str]:
        for report in self.repository.list_reports(kb_id):
            if report.community_id == community_id:
                return report.summary
        return None


def resolve_graph_scope(query: str, scope: str = "auto") -> str:
    """Resolve local/global/auto graph search scope deterministically."""

    normalized_scope = str(scope or "auto").strip().lower()
    if normalized_scope not in {"local", "global", "auto"}:
        raise ValueError("scope must be one of: local, global, auto")
    if normalized_scope != "auto":
        return normalized_scope
    query_lower = str(query or "").lower()
    chinese_global_hints = {
        "\u6574\u4f53",
        "\u603b\u4f53",
        "\u4e3b\u9898",
        "\u8d8b\u52bf",
        "\u6982\u62ec",
        "\u603b\u7ed3",
        "\u91cd\u8981\u4e8b\u9879",
        "\u7ecf\u8425\u91cd\u70b9",
        "\u7ecf\u8425\u4f18\u52bf",
        "\u4e3b\u7ebf",
        "\u677f\u5757",
        "\u903b\u8f91",
        "\u5f52\u7eb3",
        "\u652f\u6491",
        "\u957f\u671f\u53d1\u5c55",
        "\u8bf4\u660e",
        "\u4f53\u73b0",
        "\u610f\u5473",
        "\u6210\u679c",
        "\u5f71\u54cd",
        "\u8868\u660e",
    }
    if any(hint in query_lower for hint in chinese_global_hints):
        return "global"
    if any(hint in query_lower for hint in GLOBAL_SCOPE_HINTS):
        return "global"
    return "local"


def _validate_query(query: str, kb_id: str, top_k: int) -> List[str]:
    warnings = []
    if not isinstance(query, str) or not query.strip():
        warnings.append("query must be a non-empty string")
    if not isinstance(kb_id, str) or not kb_id.strip():
        warnings.append("kb_id must be a non-empty string")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        warnings.append("top_k must be a positive integer")
    return warnings


def _query_tokens(query: str) -> List[str]:
    tokens: List[str] = []
    for token in re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]+", query.lower()):
        if not token or token in STOPWORDS:
            continue
        tokens.append(token)
        if re.fullmatch(r"[\u4e00-\u9fff]+", token) and len(token) > 2:
            for size in (2, 3, 4):
                for index in range(0, len(token) - size + 1):
                    gram = token[index : index + size]
                    if gram not in STOPWORDS:
                        tokens.append(gram)
    seen = set()
    unique = []
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        unique.append(token)
    return unique


def _query_intent(query: str) -> set[str]:
    text = str(query or "").lower()
    intent = set()
    if any(marker in text for marker in {"\u6d41\u7a0b", "\u6b65\u9aa4", "\u73af\u8282", "process", "step"}):
        intent.add("process")
    if any(marker in text for marker in {"\u6e20\u9053", "\u4e3b\u4f53", "\u9500\u552e", "channel"}):
        intent.add("channel")
    if any(marker in text for marker in {"\u5305\u62ec", "\u5305\u542b", "\u54ea\u4e9b", "include", "contain"}):
        intent.add("contains")
    if any(marker in text for marker in {"\u5173\u7cfb", "\u4e4b\u95f4", "\u5148\u540e", "relationship", "relation"}):
        intent.add("relationship")
    return intent


def _entity_score(
    entity: GraphEntity, tokens: Sequence[str], query: str
) -> tuple[float, str, str]:
    haystacks = [
        ("name", entity.name),
        ("description", entity.description),
        ("type", entity.type),
    ]
    normalized_query = query.strip().lower()
    score = 0.0
    reasons = []
    matched_text = ""
    for label, text in haystacks:
        text_lower = (text or "").lower()
        if not text_lower:
            continue
        if normalized_query and normalized_query in text_lower:
            score += 2.0
            reasons.append(f"query_substring_in_{label}")
            matched_text = matched_text or text
        if text_lower and len(text_lower) >= 2 and text_lower in normalized_query:
            score += 2.0
            reasons.append(f"{label}_substring_in_query")
            matched_text = matched_text or text
        matched_tokens = [token for token in tokens if token in text_lower]
        if matched_tokens:
            score += len(matched_tokens)
            reasons.append(f"{label}_tokens={','.join(sorted(set(matched_tokens)))}")
            matched_text = matched_text or text
    return score, matched_text or entity.name, "; ".join(reasons)


def _intent_score(
    entity: GraphEntity,
    relationships: Sequence[GraphRelationship],
    intent: set[str],
) -> tuple[float, str]:
    if not intent:
        return 0.0, ""
    entity_name = (entity.name or "").lower()
    relation_types = {relationship.relation_type for relationship in relationships}
    descriptions = " ".join(
        relationship.description or relationship.relation_type
        for relationship in relationships
    ).lower()
    score = 0.0
    reasons = []
    if "process" in intent:
        if "\u6d41\u7a0b" in entity_name or "process" in entity_name:
            score += 3.0
            reasons.append("intent_process_entity")
        if "next_step" in relation_types:
            score += 2.0
            reasons.append("intent_process_next_step")
        if "\u6d41\u7a0b" in descriptions:
            score += 1.0
            reasons.append("intent_process_relationship")
    if "channel" in intent:
        if "\u6e20\u9053" in entity_name or "channel" in entity_name:
            score += 2.0
            reasons.append("intent_channel_entity")
        if "\u6e20\u9053" in descriptions:
            score += 1.0
            reasons.append("intent_channel_relationship")
    if "contains" in intent and "contains" in relation_types:
        score += 1.0
        reasons.append("intent_contains_relationship")
    if "relationship" in intent and relationships:
        score += 0.5
        reasons.append("intent_relationship_present")
    return score, "; ".join(reasons)


def _relationships_score(
    relationships: Sequence[GraphRelationship],
    tokens: Sequence[str],
    query: str,
) -> tuple[float, str, str]:
    normalized_query = query.strip().lower()
    score = 0.0
    reasons = []
    matched_text = ""
    for relationship in relationships:
        haystacks = {
            "relationship": relationship.description,
            "relation_type": relationship.relation_type,
        }
        for label, text in haystacks.items():
            text_lower = (text or "").lower()
            if not text_lower:
                continue
            if normalized_query and normalized_query in text_lower:
                score += 1.5
                reasons.append(f"query_substring_in_{label}")
                matched_text = matched_text or text
            matched_tokens = [token for token in tokens if len(token) >= 2 and token in text_lower]
            if matched_tokens:
                score += min(len(set(matched_tokens)) * 0.25, 2.0)
                reasons.append(f"{label}_tokens={','.join(sorted(set(matched_tokens))[:6])}")
                matched_text = matched_text or text
    return score, "; ".join(reasons), matched_text


def _report_score(
    report: CommunityReport, tokens: Sequence[str], query: str
) -> tuple[float, str]:
    searchable = {
        "title": report.title,
        "summary": report.summary,
        "key_entities": " ".join(report.key_entities),
        "key_relationships": " ".join(report.key_relationships),
    }
    normalized_query = query.strip().lower()
    score = 0.0
    reasons = []
    for label, text in searchable.items():
        text_lower = (text or "").lower()
        if normalized_query and normalized_query in text_lower:
            score += 2.0
            reasons.append(f"query_substring_in_{label}")
        matched_tokens = [token for token in tokens if token in text_lower]
        if matched_tokens:
            score += len(set(matched_tokens))
            reasons.append(f"{label}_tokens={','.join(sorted(set(matched_tokens)))}")
    return score, "; ".join(reasons)


def _report_quality_score(report: CommunityReport) -> int:
    labels = [
        *_split_report_title(report.title),
        *[str(entity) for entity in report.key_entities],
        *[str(relationship) for relationship in report.key_relationships],
    ]
    meaningful_labels = [
        label for label in labels if _label_quality_score(label) > 0
    ]
    if not meaningful_labels:
        return 0
    score = sum(_label_quality_score(label) for label in meaningful_labels)
    score += min(len(report.source_refs), 3)
    score += min(len(report.key_relationships), 2)
    return score


def _split_report_title(title: str) -> List[str]:
    cleaned = _clean_report_label(title).replace("Community:", "")
    return [part.strip() for part in cleaned.split(",") if part.strip()]


def _clean_report_label(value: Any) -> str:
    text = str(value or "").replace("Community:", "").strip()
    text = text.strip(" `\"'.,;:()[]{}")
    return " ".join(text.split())


def _label_quality_score(value: Any) -> int:
    text = _clean_report_label(value)
    if not text:
        return 0

    weak_labels = {
        "\u4e00",
        "\u4e8c",
        "\u4e09",
        "\u56db",
        "\u4e94",
        "\u516d",
        "\u4e03",
        "\u516b",
        "\u4e5d",
        "\u5341",
        "\u5f52",
        "\u4e0a\u5e02",
        "\u4e0a\u5e02\u516c\u53f8\u80a1\u4e1c\u7684",
    }
    if text in weak_labels or len(text) <= 1:
        return 0
    if text.endswith("\u7684") and len(text) <= 12:
        return 0
    if re.fullmatch(r"[A-Z]{1,5}", text) and text not in {"ESG"}:
        return 0

    strong_keywords = {
        "\u54c1\u8d28",
        "\u8d28\u91cf",
        "\u54c1\u724c",
        "\u5e02\u573a",
        "\u5de5\u827a",
        "\u751f\u4ea7",
        "\u6587\u5316",
        "ESG",
        "\u6cbb\u7406",
        "\u5e02\u503c",
        "\u73af\u5883",
        "\u751f\u6001",
        "\u8305\u53f0\u9152",
        "\u4e3b\u8981\u4e1a\u52a1",
    }
    score = 1
    if any(keyword in text for keyword in strong_keywords):
        score += 4
    if len(text) >= 4:
        score += 1
    return score


def _neighbor_entities(
    entity: GraphEntity,
    relationships: Sequence[GraphRelationship],
    all_entities: Sequence[GraphEntity],
) -> List[GraphEntity]:
    entity_by_id = {item.id: item for item in all_entities}
    neighbors = []
    seen = set()
    for relationship in relationships:
        neighbor_id = (
            relationship.target_entity_id
            if relationship.source_entity_id == entity.id
            else relationship.source_entity_id
        )
        if neighbor_id in seen or neighbor_id not in entity_by_id:
            continue
        seen.add(neighbor_id)
        neighbors.append(entity_by_id[neighbor_id])
    return sorted(neighbors, key=lambda item: item.name.lower())


def _local_path(
    query: str,
    entity: GraphEntity,
    relationships: Sequence[GraphRelationship],
    neighbors: Sequence[GraphEntity],
    source_refs: Sequence[GraphSourceRef],
) -> List[str]:
    path = [f"query:{query}", f"entity:{entity.name}"]
    if relationships:
        first_relationship = relationships[0]
        neighbor = next(
            (
                item
                for item in neighbors
                if item.id
                in {
                    first_relationship.source_entity_id,
                    first_relationship.target_entity_id,
                }
            ),
            None,
        )
        path.append(f"relationship:{first_relationship.relation_type}")
        if neighbor:
            path.append(f"neighbor:{neighbor.name}")
    if source_refs:
        ref = source_refs[0]
        path.append(f"source_chunk:{ref.document_id}/{ref.chunk_id}")
    return path


def _rag_chunk_from_local_hit(hit: GraphSearchHit, *, rank: int) -> RAGChunk:
    ref = hit.source_refs[0] if hit.source_refs else None
    content = _local_hit_content(hit)
    return RAGChunk(
        chunk_id=(ref.chunk_id if ref else f"graph_local_{hit.entity_id}"),
        content=content,
        source=RAGSource(
            document_id=(ref.document_id if ref else hit.entity_id),
            chunk_index=0,
            source_path=(ref.filename if ref else None),
            title=(ref.section or ref.filename if ref else hit.entity_name),
            page=(ref.page if ref else None),
            score=hit.score,
            metadata={
                "graph_scope": "local",
                "entity_id": hit.entity_id,
                "entity_name": hit.entity_name,
                "relationship_id": (
                    hit.relationships[0].id if hit.relationships else None
                ),
                "path": list(hit.path),
                "rank": rank,
                "score_reason": hit.metadata.get("score_reason"),
            },
        ),
    )


def _rag_chunk_from_global_hit(hit: GraphGlobalSearchHit, *, rank: int) -> RAGChunk:
    ref = hit.source_refs[0] if hit.source_refs else None
    return RAGChunk(
        chunk_id=(ref.chunk_id if ref else f"graph_global_{hit.report_id}"),
        content=hit.summary,
        source=RAGSource(
            document_id=(ref.document_id if ref else hit.community_id),
            chunk_index=0,
            source_path=(ref.filename if ref else None),
            title=hit.title,
            page=(ref.page if ref else None),
            score=hit.score,
            metadata={
                "graph_scope": "global",
                "community_id": hit.community_id,
                "report_id": hit.report_id,
                "rank": rank,
                "score_reason": hit.metadata.get("score_reason"),
            },
        ),
    )


def _local_hit_content(hit: GraphSearchHit) -> str:
    relationship_text = "; ".join(
        relationship.description or relationship.relation_type
        for relationship in hit.relationships
    )
    neighbor_text = ", ".join(entity.name for entity in hit.neighbor_entities)
    return (
        f"Entity: {hit.entity_name}. "
        f"Relationships: {relationship_text or 'none'}. "
        f"Neighbors: {neighbor_text or 'none'}."
    )


def _dedupe_source_refs(refs: Iterable[GraphSourceRef]) -> List[GraphSourceRef]:
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
    "GraphGlobalSearchHit",
    "GraphGlobalSearchResult",
    "GraphLocalSearchResult",
    "GraphRetriever",
    "GraphSearchHit",
    "resolve_graph_scope",
]
