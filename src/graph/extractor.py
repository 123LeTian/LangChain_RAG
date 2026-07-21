"""Deterministic graph entity and relationship extraction."""

from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Mapping, Tuple

from src.models.graph import (
    GraphEntity,
    GraphExtractionResult,
    GraphRelationship,
    GraphSourceRef,
)


RELATION_PATTERNS = [
    (re.compile(r"(.+?)\s+(?:is|are)\s+(?:a|an|the)?\s*(.+)", re.I), "is_a"),
    (re.compile(r"(.+?)\s+belongs\s+to\s+(.+)", re.I), "belongs_to"),
    (re.compile(r"(.+?)\s+depends\s+on\s+(.+)", re.I), "depends_on"),
    (re.compile(r"(.+?)\s+calls\s+(.+)", re.I), "calls"),
    (re.compile(r"(.+?)\s+contains\s+(.+)", re.I), "contains"),
    (re.compile(r"(.+?)\s+relates\s+to\s+(.+)", re.I), "related_to"),
    (re.compile(r"(.+?)\s*是\s*(.+)"), "is_a"),
    (re.compile(r"(.+?)\s*属于\s*(.+)"), "belongs_to"),
    (re.compile(r"(.+?)\s*依赖\s*(.+)"), "depends_on"),
    (re.compile(r"(.+?)\s*调用\s*(.+)"), "calls"),
    (re.compile(r"(.+?)\s*包含\s*(.+)"), "contains"),
    (re.compile(r"(.+?)\s*与\s*(.+?)\s*相关"), "related_to"),
]

ENTITY_PATTERN = re.compile(
    r"`([^`]+)`|([A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*){0,3})"
)


class GraphExtractor(ABC):
    """Base interface for graph extraction implementations."""

    @abstractmethod
    def extract_from_chunk(self, chunk: Any) -> GraphExtractionResult:
        """Extract graph objects from one chunk."""

    def extract_from_chunks(self, chunks: Iterable[Any]) -> GraphExtractionResult:
        entities: Dict[str, GraphEntity] = {}
        relationships: Dict[str, GraphRelationship] = {}
        warnings: List[str] = []
        for chunk in chunks:
            try:
                result = self.extract_from_chunk(chunk)
            except Exception as exc:
                warnings.append(f"chunk extraction failed: {type(exc).__name__}: {exc}")
                continue
            warnings.extend(result.warnings)
            for entity in result.entities:
                entities[entity.id] = _merge_entity(entities.get(entity.id), entity)
            for relationship in result.relationships:
                relationships[relationship.id] = _merge_relationship(
                    relationships.get(relationship.id), relationship
                )
        return GraphExtractionResult(
            entities=list(entities.values()),
            relationships=list(relationships.values()),
            warnings=warnings,
        )


class RuleBasedGraphExtractor(GraphExtractor):
    """Small deterministic extractor for D3 Graph Index MVP."""

    def extract_from_chunk(self, chunk: Any) -> GraphExtractionResult:
        try:
            chunk_data = _chunk_data(chunk)
            source_ref = _source_ref(chunk_data)
        except ValueError as exc:
            return GraphExtractionResult(warnings=[str(exc)])

        text = chunk_data["text"]
        kb_id = chunk_data["kb_id"]
        entities: Dict[str, GraphEntity] = {}
        relationships: Dict[str, GraphRelationship] = {}

        for sentence in _sentences(text):
            relation = _extract_relation(sentence)
            if relation is None:
                for name in _extract_entity_mentions(sentence):
                    entity = _entity(name, source_ref, kb_id)
                    entities[entity.id] = _merge_entity(entities.get(entity.id), entity)
                continue

            source_name, relation_type, target_name = relation
            source_entity = _entity(source_name, source_ref, kb_id)
            target_entity = _entity(target_name, source_ref, kb_id)
            entities[source_entity.id] = _merge_entity(
                entities.get(source_entity.id), source_entity
            )
            entities[target_entity.id] = _merge_entity(
                entities.get(target_entity.id), target_entity
            )
            relationship = _relationship(
                source_entity.id,
                target_entity.id,
                relation_type,
                source_ref,
                kb_id,
                source_name,
                target_name,
            )
            relationships[relationship.id] = _merge_relationship(
                relationships.get(relationship.id), relationship
            )

        return GraphExtractionResult(
            entities=list(entities.values()),
            relationships=list(relationships.values()),
        )


MockGraphExtractor = RuleBasedGraphExtractor


def stable_entity_id(name: str, entity_type: str = "concept") -> str:
    normalized = normalize_entity_name(name)
    digest = hashlib.sha1(f"{entity_type}:{normalized}".encode("utf-8")).hexdigest()
    return f"ent_{digest[:16]}"


def stable_relationship_id(
    source_entity_id: str, target_entity_id: str, relation_type: str
) -> str:
    raw = f"{source_entity_id}:{relation_type}:{target_entity_id}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()
    return f"rel_{digest[:16]}"


def normalize_entity_name(name: str) -> str:
    normalized = re.sub(r"\s+", " ", name.strip(" `\"'，,。.;；:：()[]{}")).strip()
    return normalized.lower()


def _chunk_data(chunk: Any) -> Dict[str, Any]:
    if isinstance(chunk, Mapping):
        data = dict(chunk)
    elif hasattr(chunk, "to_dict"):
        data = chunk.to_dict()
    else:
        data = {
            key: getattr(chunk, key)
            for key in ("id", "document_id", "kb_id", "text", "metadata")
            if hasattr(chunk, key)
        }

    chunk_id = data.get("id") or data.get("chunk_id")
    document_id = data.get("document_id")
    kb_id = data.get("kb_id") or data.get("knowledge_base_id") or "default"
    text = data.get("text") or data.get("content")
    metadata = data.get("metadata") or {}
    if not isinstance(metadata, Mapping):
        metadata = {}
    if not chunk_id or not document_id:
        raise ValueError("chunk-like object must include document_id and chunk id")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"chunk {chunk_id}: text must be a non-empty string")
    return {
        "chunk_id": str(chunk_id),
        "document_id": str(document_id),
        "kb_id": str(kb_id),
        "text": text.strip(),
        "metadata": dict(metadata),
    }


def _source_ref(chunk_data: Mapping[str, Any]) -> GraphSourceRef:
    metadata = dict(chunk_data.get("metadata") or {})
    return GraphSourceRef(
        document_id=str(chunk_data["document_id"]),
        chunk_id=str(chunk_data["chunk_id"]),
        filename=_optional_str(metadata.get("filename") or metadata.get("source")),
        page=metadata.get("page") if isinstance(metadata.get("page"), int) else None,
        section=_optional_str(metadata.get("section")),
        quote=str(chunk_data["text"])[:500],
        metadata=metadata,
    )


def _sentences(text: str) -> List[str]:
    return [
        sentence.strip()
        for sentence in re.split(r"[\n。.!?！？;；]+", text)
        if sentence.strip()
    ]


def _extract_relation(sentence: str) -> Tuple[str, str, str] | None:
    for pattern, relation_type in RELATION_PATTERNS:
        match = pattern.fullmatch(sentence.strip())
        if not match:
            continue
        left = _clean_entity_text(match.group(1))
        right = _clean_entity_text(match.group(2))
        if left and right and normalize_entity_name(left) != normalize_entity_name(right):
            return left, relation_type, right
    return None


def _extract_entity_mentions(sentence: str) -> List[str]:
    names = []
    for match in ENTITY_PATTERN.finditer(sentence):
        name = _clean_entity_text(match.group(1) or match.group(2))
        if name:
            names.append(name)
    return names


def _clean_entity_text(value: str) -> str:
    text = value.strip()
    text = re.sub(r"^(?:the|a|an)\s+", "", text, flags=re.I)
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" `\"'，,。.;；:：()[]{}")
    if len(text) > 80:
        text = text[:80].strip()
    return text


def _entity(name: str, source_ref: GraphSourceRef, kb_id: str) -> GraphEntity:
    normalized = normalize_entity_name(name)
    display_name = name.strip()
    return GraphEntity(
        id=stable_entity_id(display_name),
        name=display_name,
        type="concept",
        description=f"Entity extracted from chunk {source_ref.chunk_id}.",
        source_refs=[source_ref],
        metadata={"normalized_name": normalized, "kb_id": kb_id},
    )


def _relationship(
    source_entity_id: str,
    target_entity_id: str,
    relation_type: str,
    source_ref: GraphSourceRef,
    kb_id: str,
    source_name: str,
    target_name: str,
) -> GraphRelationship:
    return GraphRelationship(
        id=stable_relationship_id(source_entity_id, target_entity_id, relation_type),
        source_entity_id=source_entity_id,
        target_entity_id=target_entity_id,
        relation_type=relation_type,
        description=f"{source_name} {relation_type} {target_name}",
        weight=1.0,
        source_refs=[source_ref],
        metadata={"kb_id": kb_id},
    )


def _merge_entity(
    existing: GraphEntity | None, incoming: GraphEntity
) -> GraphEntity:
    if existing is None:
        return incoming
    refs = _dedupe_source_refs([*existing.source_refs, *incoming.source_refs])
    metadata = {**existing.metadata, **incoming.metadata}
    return existing.model_copy(update={"source_refs": refs, "metadata": metadata})


def _merge_relationship(
    existing: GraphRelationship | None, incoming: GraphRelationship
) -> GraphRelationship:
    if existing is None:
        return incoming
    refs = _dedupe_source_refs([*existing.source_refs, *incoming.source_refs])
    metadata = {**existing.metadata, **incoming.metadata}
    return existing.model_copy(
        update={
            "weight": existing.weight + incoming.weight,
            "source_refs": refs,
            "metadata": metadata,
        }
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


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


__all__ = [
    "GraphExtractor",
    "MockGraphExtractor",
    "RuleBasedGraphExtractor",
    "normalize_entity_name",
    "stable_entity_id",
    "stable_relationship_id",
]
